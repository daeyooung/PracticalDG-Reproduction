# coding=utf-8
"""CLIPBase: CLIP(ViT-B/16)의 zero-shot 예측 확률을 soft label 로 삼아
ResNet 학생 모델을 학습하는 베이스라인 (PracticalDG 논문의 'CLIP Base')."""
import torch
import clip

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from utils.daml_util import manual_CE
from utils.class_names import clip_text_inputs


class CLIPBase(Algorithm):
    def __init__(self, args):
        super(CLIPBase, self).__init__(args)
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier
        )
        self.args = args
        self.device = torch.device(f'cuda:{self.args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.model, _ = clip.load('ViT-B/16', self.device)   # teacher (frozen)
        self.bz = args.batch_size
        self.num_classes = args.num_classes
        # 데이터셋별 known 클래스 이름 -> CLIP 텍스트 토큰 (한 번만 만든다)
        self.text_inputs = clip_text_inputs(args.dataset, self.device, args.num_classes)

    def torch_cosine_similarity(self, features1, features2):
        norm1 = torch.norm(features1, dim=-1).reshape(features1.shape[0], 1)
        norm2 = torch.norm(features2, dim=-1).reshape(1, features2.shape[0])
        end_norm = torch.mm(norm1.to(torch.double), norm2.to(torch.double))
        cos = torch.mm(features1.to(torch.double), features2.T.to(torch.double)) / end_norm
        return cos

    def clip_soft_labels(self, x):
        """CLIP zero-shot 확률 (N, num_classes) 과 정규화된 텍스트 특징을 돌려준다."""
        with torch.no_grad():
            image_features = self.model.encode_image(x)
            text_features = self.model.encode_text(self.text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100 * image_features @ text_features.T).softmax(dim=-1)
        return similarity.float().detach(), text_features

    def update(self, minibatches, opt, sch):
        # data preparation
        all_x = torch.cat([data[0].to(self.device).float() for data in minibatches])
        opt.zero_grad()

        # teacher: CLIP soft label
        similarity, _ = self.clip_soft_labels(all_x)

        # student: ResNet feature -> linear classifier
        features = self.featurizer(all_x)
        class_logits = self.classifier(features)
        class_loss = manual_CE(class_logits, similarity)
        loss = class_loss

        loss.backward()
        opt.step()

        if sch:
            sch.step()

        return {"class": class_loss.item()}

    def predict(self, x):
        features = self.featurizer(x)
        output = self.classifier(features)
        return output

    def base(self, x):
        """CLIP zero-shot 예측 (학생 모델 없이 teacher 만 사용)."""
        similarity, _ = self.clip_soft_labels(x)
        return similarity
