#!/bin/bash

is_different_class_space=(0)   #  1 2 3
class_space_num=${#is_different_class_space[@]}
is_data_aug=yes
loader_name='daml_loader'
dataset='office-home'
data_dir='/Dataset/OfficeHome'
# dataset='PACS'
# data_dir='/Dataset/PACS/'

algorithms=('CLIPBase')   # 'SCIPD',
algorithm_num=${#algorithms[@]}

net='resnet18'
lr=1e-3
gpu_id=0
max_epoch=50
domain_num=4       
seed=0


for ((test_envs=0; test_envs < 4; test_envs++)); do
    for ((i=0; i < $algorithm_num; i++)); do
        for ((j=0; j < $class_space_num; j++)); do
            output='./output/overall/'${dataset}'/'${dataset}'_class_space'${is_different_class_space[j]}'/'${algorithms[i]}'/test_envs_'${test_envs}'/seed_'${seed}
            # output='./output/'${loader_name}'/lr_'${lr}'_dataaug_'${is_data_aug}'/'${dataset}'/'${dataset}'_class_space'${is_different_class_space[j]}'/'${algorithms[i]}'/test_envs_'${test_envs}'/seed_'${seed}
            # train
            python train.py --seed $seed --lr $lr --data_dir $data_dir --domain_num $domain_num --max_epoch $max_epoch --net $net --output $output \
            --test_envs $test_envs --dataset $dataset --algorithm ${algorithms[i]} --gpu_id $gpu_id \
            --is_different_class_space $is_different_class_space --is_data_aug $is_data_aug --loader_name $loader_name --is_different_class_space ${is_different_class_space[j]}
            # acc and auroc
            python eval.py --seed $seed --lr $lr --data_dir $data_dir --domain_num $domain_num --max_epoch $max_epoch --net $net --output $output \
            --test_envs $test_envs --dataset $dataset --algorithm ${algorithms[i]} --gpu_id $gpu_id \
            --is_different_class_space $is_different_class_space --loader_name $loader_name --is_data_aug $is_data_aug --is_different_class_space ${is_different_class_space[j]}
            # h-score
            python eval_hscore.py --seed $seed --lr $lr --data_dir $data_dir --domain_num $domain_num --max_epoch $max_epoch --net $net --output $output \
            --test_envs $test_envs --dataset $dataset --algorithm ${algorithms[i]} --gpu_id $gpu_id \
            --is_different_class_space $is_different_class_space --loader_name $loader_name --is_data_aug $is_data_aug --is_different_class_space ${is_different_class_space[j]}
        done
    done
done