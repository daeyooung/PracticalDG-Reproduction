#!/bin/bash

is_different_class_space=(0 1 2 3)
class_space_num=${#is_different_class_space[@]}
is_data_aug=yes
loader_name='daml_loader'
dataset='office-home'
data_dir='/Dataset/OfficeHome'

algorithms=('SCIPD') 
algorithm_num=${#algorithms[@]}

net='resnet18'
task='img_dg'
lr=1e-3
gpu_id=0
max_epoch=50
steps_per_epoch=100
domain_num=4


for ((seed=0; seed < 1; seed++)); do
    for ((test_envs=0; test_envs < 1; test_envs++)); do
        for ((i=0; i < $algorithm_num; i++)); do
            for ((j=0; j < $class_space_num; j++)); do
                output='./output/'${loader_name}'/lr_'${lr}'_dataaug_'${is_data_aug}'/'${dataset}'_class_space'${is_different_class_space[j]}'/'${algorithms[i]}'/test_envs_'${test_envs}'/seed_'${seed}
                # h-score
                python eval_hscore_val.py --seed $seed --lr $lr --data_dir $data_dir --domain_num $domain_num --max_epoch $max_epoch --net $net --task $task --output $output \
                --test_envs $test_envs --dataset $dataset --algorithm ${algorithms[i]} --steps_per_epoch $steps_per_epoch --gpu_id $gpu_id \
                --is_different_class_space $is_different_class_space --loader_name $loader_name --is_data_aug $is_data_aug --is_different_class_space ${is_different_class_space[j]}
            done
        done
    done
done