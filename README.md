# Reproducing "PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization [CVPR.2024]"

This repository contains a reproduction study of the paper:

> **"[PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization](https://arxiv.org/abs/2404.09011)"**, Zining Chen et al., CVPR, 2024.

We utilize the [official implementation](https://github.com/znchen666/HDG) provided by the authors and extend it for additional analysis and experiments.

---

## Requirements
```
Python 3.7.11+
Pytorch 1.8.0+
```

## Data Preparation
Download the dataset [PACS](https://github.com/lingeringlight/DomainDrop?tab=readme-ov-file), [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html).
Create a `Dataset` folder and insert the downloaded dataset.
```
/PACS
├── Domain1
      ├── cat
      ├── ......
├── Domain2
      ├── cat
      ├── ......
/OfficeHome
├── Domain1
      ├── cat
      ├── ......
├── Domain2
      ├── cat
      ├── ......

```

## Scope of reproducibility
We focus on the following reproduction parts.

- **Model Performance**: Measure and compare the performance of the proposed method and the CLIP Base method (Table 1,2 / Fig 4)
- **Effectiveness of Loss component**: Measure and compare the proposed loss component performance (Table 5)
- **Sensitivity of Hyper-parameter setting**: Check model sensitivity to hyperparameter changes (Fig 5)
- **Realistic Treshold setting**: Set the threshold on the validation set and measure model performance
- In this study, we used the OfficeHome and PACS datasets (mainly OfficeHome). Except for the "Model Performance" section, all other comparative experiments were conducted on a single test domain.


## Run to check Model Performance
```
bash scripts/run.sh
```
Run the command above to save the model results for all domain. If you want to check for single domain performance, Adjust `test_env` in sh file.
To use a different dataset, change the `dataset` and `data_dir` arguments in the code. 
The results are saved in the `output` folder. The file path follows this structure: Dataset/H(class space)/domain(test_env)
Performance metrics include accuracy and AUROC for validation/test data, and H-score for test data. These metrics are saved in done.txt, eval_h_score, and eval_h_score.txt.

```
python ./visualize/compare_clipbase_scipd.py \
  --root_dir ./output/overall/office-home \   
  --plot_dir ./output/overall/office-home \
  --csv_path ./output/overall/office-home/officehome_compare.csv
```
Run the command above to visualize the performance on the target domain. 
You can compare the performance of `CLIPBase` and the `proposed method` across different H (class spaces) and domains.


## Run to check Effectiveness of Loss component
```
bash scripts/run_loss.sh loss_type
```
Run the command above to check the performance of models using the specific loss terms proposed in the paper. 
You can compare the performance of CLIPBase, I-PD, SI-PD, and SCI-PD.
Simply input the target loss type and execute.
For CLIPBase, enter any loss type and set algorithms to CLIPBase in the .sh file.



## Run to check Sensitivity of Hyper-parameter setting
```
bash scripts/run_val.sh hyper
```
Run the command above to check model robustness by adjusting the hyperparameters. 
This script runs experiments for `alpha`, `beta`, and `tau` within the ranges suggested in the paper and saves the results.
Specify the target hyperparameter (hyper) and run.
```
python ./visualize/visualize_hyper.py \
  --root_dir ./output/ablation/hyper/alpha \
  --hyper_name alpha \
  --acc_fig_path ./output/ablation/hyper/alpha/alpha_acc.png \
  --hscore_fig_path ./output/ablation/hyper/alpha/alpha_hscore.png \
  --csv_path ./output/ablation/hyper/alpha/alpha_summary.csv \
  --no_show
```
Run the command above to visualize the H-score for each hyperparameter, similar to Figure 5 in the paper.
Replace `alpha` in the command above with the target hyperparameter for visualization.


## Run for Realistic Treshold setting
```
bash scripts/run_val.sh
```
Run the command above to measure performance using a threshold set by the validation set for unknown detection.
The results are saved in /eval/`eval_h_score.txt`. This file contains the target domain performance for each threshold.
```
python ./visualize/threshold_val_test_user.py \
  --base_dir ./output/overall/office-home \
  --val_line_num 3 \
  --no_show
```
Run the above to visualize the performance difference between the validation-based threshold and the test domain threshold.


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
