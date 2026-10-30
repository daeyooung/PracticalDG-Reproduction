# coding=utf-8
"""SCI-PD: Score / Class / Instance Perturbation Distillation (PracticalDG, CVPR 2024).

CLIP(ViT-B/16) teacher 의 zero-shot 확률을 세 가지 섭동(perturbation)으로 보정하여
ResNet 학생 모델에 증류한다.
  - IP  : instance perturbation  (CLIP 확신도가 낮은 샘플에 큰 가중치)
  - SI  : + score perturbation   (CLIP 이 틀린 샘플의 soft label 을 정답 쪽으로 보정)
  - SCI : + class perturbation   (분류기 가중치와 텍스트 임베딩 간 클래스 관계 정렬)
--loss_type 으로 IP / SI / SCI 를 선택한다 (논문 Table 5 ablation).
"""
import torch
import torch.nn.functional as F
import clip

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from utils.daml_util import manual_CE, weighted_manual_CE
from utils.class_names import clip_text_inputs


class SCIPD(Algorithm):
    def __init__(self, args):
        super(SCIPD, self).__init__(args)
        self.featurizer = get_fea(args)    # resnet
        self.classifier = common_network.feat_classifier_new(
            args.num_classes, self.featurizer.in_features, args.classifier
        )
        self.args = args
        self.device = torch.device(f'cuda:{self.args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.model, _ = clip.load('ViT-B/16', self.device)   # teacher (frozen)
        self.bz = args.batch_size
        self.num_classes = args.num_classes
        self.loss_type = args.loss_type

        # 데이터셋별 known 클래스 이름 -> CLIP 텍스트 토큰 (한 번만 만든다)
        self.text_inputs = clip_text_inputs(args.dataset, self.device, args.num_classes)

        # 분류기 마지막 층을 CLIP 텍스트 임베딩으로 초기화
        with torch.no_grad():
            text_features = self.model.encode_text(self.text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.classifier.fc.weight.data.copy_(text_features.float())
        # featurizer.in_features: resnet image embedding (512 dim) -> not used
        self.register_buffer(
            "proto_feats",
            torch.zeros((args.num_classes, self.featurizer.in_features), requires_grad=False),
        )

    def torch_cosine_similarity(self, features1, features2):
        norm1 = torch.norm(features1, dim=-1).reshape(features1.shape[0], 1)
        norm2 = torch.norm(features2, dim=-1).reshape(1, features2.shape[0])
        end_norm = torch.mm(norm1.to(torch.double), norm2.to(torch.double))
        cos = torch.mm(features1.to(torch.double), features2.T.to(torch.double)) / end_norm
        return cos

    def update(self, minibatches, opt, sch):
        loss_type = self.loss_type
        # data preparation
        all_x = torch.cat([data[0].to(self.device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(self.device).long() for data in minibatches])

        net_opt = opt[0]    # feature weight
        cls_opt = opt[1]    # classifier weight
        net_opt.zero_grad()
        cls_opt.zero_grad()

        # teacher: CLIP zero-shot 확률 (image's logit)
        with torch.no_grad():
            image_features = self.model.encode_image(all_x)
            text_features = self.model.encode_text(self.text_inputs)
            text_sim = self.torch_cosine_similarity(text_features, text_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1).float()

        # instance perturbation: sample selection standard
        weight_ori = torch.max(similarity, dim=1)[0]
        weight_inverse = 1 / torch.pow(weight_ori, self.args.hyper1)   # 0.8

        # student feature output
        features = self.featurizer(all_x)   # resnet
        class_logits = self.classifier(features)

        if loss_type == 'IP':
            si_loss = weighted_manual_CE(class_logits, similarity, weight_inverse)
            loss = si_loss
        else:
            # score perturbation: fix wrong samples
            similarity_si = similarity.clone()
            predict_label = torch.argmax(similarity_si, dim=1)
            wrong_index = torch.nonzero(predict_label != all_y).squeeze(1)
            if wrong_index.numel() > 0:
                wrong_onehot = F.one_hot(all_y[wrong_index], num_classes=self.args.num_classes).float()
                wrong_similarity = similarity_si[wrong_index]
                max_similarity, _ = wrong_similarity.max(dim=1, keepdim=False)
                similarity_si[wrong_index] = F.softmax(
                    (wrong_onehot * max_similarity.unsqueeze(1) + wrong_similarity) / self.args.hyper2,
                    dim=1,
                )
            si_loss = weighted_manual_CE(class_logits, similarity_si, weight_inverse)

            if loss_type == 'SI':
                loss = si_loss
            elif loss_type == 'SCI':
                # class perturbation: text-text 유사도와 text-classifier 유사도 정렬
                text_features_norm = (text_features / text_features.norm(p=2, dim=-1, keepdim=True)).float()
                cls_weight = self.classifier.fc.weight
                cls_weight_norm = cls_weight / cls_weight.norm(p=2, dim=-1, keepdim=True)
                textcls_sim = torch.matmul(text_features_norm, cls_weight_norm.t())
                text_sim_dev = text_sim.to(self.device)
                contrastive_loss = (
                    manual_CE(torch.softmax(textcls_sim, dim=1), torch.softmax(text_sim_dev, dim=1))
                    + manual_CE(torch.softmax(textcls_sim.t(), dim=1), torch.softmax(text_sim_dev.t(), dim=1))
                ) / 2
                loss = si_loss + self.args.hyper3 * contrastive_loss
            else:
                raise ValueError(f"unknown loss_type: {loss_type} (IP / SI / SCI)")

        loss.backward()
        net_opt.step()
        cls_opt.step()

        if sch:
            sch[0].step()
            sch[1].step()

        if loss_type == 'SCI':
            return {"class": si_loss.item()}
        return {"class": loss.item()}

    def predict(self, x):
        features = self.featurizer(x)
        output = self.classifier(features)
        return output

    def base(self, x):
        """CLIP zero-shot 예측 (학생 모델 없이 teacher 만 사용)."""
        with torch.no_grad():
            image_features = self.model.encode_image(x)
            text_features = self.model.encode_text(self.text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
        return similarity
