#!/bin/bash

# export OMP_NUM_THREADS=4  
# export MKL_NUM_THREADS=4

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

n_workers=12

# for hyper1 in 0.1 0.2 0.4 1; do
for ((seed=0; seed < 1; seed++)); do
    for ((test_envs=0; test_envs < 1; test_envs++)); do
        for ((i=0; i < $algorithm_num; i++)); do
            for ((j=0; j < $class_space_num; j++)); do
                output='./output/ablation/hyper/alpha_'${hyper1}'/'${loader_name}'/lr_'${lr}'_dataaug_'${is_data_aug}'/'${dataset}'_class_space'${is_different_class_space[j]}'/'${algorithms[i]}'/test_envs_'${test_envs}'/seed_'${seed}
                # train
                python train.py --seed $seed --lr $lr --data_dir $data_dir --domain_num $domain_num --max_epoch $max_epoch --net $net --task $task --output $output \
                --test_envs $test_envs --dataset $dataset --algorithm ${algorithms[i]} --steps_per_epoch $steps_per_epoch --gpu_id $gpu_id \
                --is_different_class_space $is_different_class_space --is_data_aug $is_data_aug --loader_name $loader_name --is_different_class_space ${is_different_class_space[j]} --N_WORKERS $n_workers --hyper1 $hyper1
                # acc and auroc
                python eval.py --seed $seed --lr $lr --data_dir $data_dir --domain_num $domain_num --max_epoch $max_epoch --net $net --task $task --output $output \
                --test_envs $test_envs --dataset $dataset --algorithm ${algorithms[i]} --steps_per_epoch $steps_per_epoch --gpu_id $gpu_id \
                --is_different_class_space $is_different_class_space --loader_name $loader_name --is_data_aug $is_data_aug --is_different_class_space ${is_different_class_space[j]} --N_WORKERS $n_workers --hyper1 $hyper1
                # h-score
                python eval_hscore.py --seed $seed --lr $lr --data_dir $data_dir --domain_num $domain_num --max_epoch $max_epoch --net $net --task $task --output $output \
                --test_envs $test_envs --dataset $dataset --algorithm ${algorithms[i]} --steps_per_epoch $steps_per_epoch --gpu_id $gpu_id \
                --is_different_class_space $is_different_class_space --loader_name $loader_name --is_data_aug $is_data_aug --is_different_class_space ${is_different_class_space[j]} --N_WORKERS $n_workers --hyper1 $hyper1
            done
        done
    done
done
# done