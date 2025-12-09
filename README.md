# Reproducing "PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization [CVPR.2024]"

This repository contains a reproduction study of the paper:

> **"PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization"**, Zining Chen et al., CVPR, 2024.

If available, we utilize the official implementation provided by the authors and extend it for additional analysis and experiments.

---

## Requirements
```
Python 3.7.11+
Pytorch 1.8.0+
```

## Data Preparation
Download the dataset PACS, OfficeHome.

Arrange data with the following structure:
```
./Dataset
├── Domain1
      ├── cat
      ├── ......
├── Domain2
      ├── cat
      ├── ......
├── Domain3
      ├── cat
      ├── ......
├── Domain4
      ├── cat
      ├── ......
├── image_list
      List for each dataset is provided in ./image_list 
```

## Scope of reproducibility
We focus on the following reproduction parts.

- **Claim 1**: (예: Proposed method outperforms baseline methods on DATASET under SCENARIO. → e.g., Table 1)
- **Claim 2**: (예: Certain component/regularizer significantly improves performance. → e.g., Table 2 / Fig. 3)
- **Claim 3**: (예: Method is robust to HYPER-PARAMETER or DATASET variations. → e.g., Fig. 4)
- **Claim 4**: (선택) Additional ablation or robustness study.


```
bash scripts/run.sh
```
Change `test_envs` to different values (e.g., 0,1,2,3) to conduct leave-one-domain-out protocol.

## Acknowledgment
We thank the authors from [OpenDG-Eval](https://github.com/shiralab/OpenDG-Eval) for reference. We modify their code to implement Hybrid Domain Generalization.

## Citation
```
@inproceedings{chen2024practicaldg,
  title={PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization},
  author={Chen, Zining and Wang, Weiqiu and Zhao, Zhicheng and Su, Fei and Men, Aidong and Meng, Hongying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23501--23511},
  year={2024}
}
```
