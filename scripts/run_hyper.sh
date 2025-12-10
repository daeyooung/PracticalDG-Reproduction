#!/bin/bash

target="$1"

if [[ "$target" == "tau" ]]; then
    vals="0.1 0.3 0.7 0.9"
    flag="--hyper2"
elif [[ "$target" == "alpha" ]]; then
    vals="0.1 0.2 0.4 0.8 1.0"
    flag="--hyper1"
elif [[ "$target" == "beta" ]]; then
    vals="0.05 0.1 0.2 0.5 1.0"
    flag="--hyper3"
else
    exit 1
fi

dataset='office-home'
data_dir='/Dataset/OfficeHome'
algorithms=('SCIPD')
class_spaces=(0 1 2 3)

net='resnet18'
lr=1e-3
gpu_id=0
max_epoch=50
domain_num=4
seed=0
test_envs=0

for val in $vals; do
    for algo in "${algorithms[@]}"; do
        for c_space in "${class_spaces[@]}"; do
            
            output="./output/ablation/hyper/${target}/${val}/${dataset}_class_space${c_space}/${algo}/test_envs_${test_envs}/seed_${seed}"
            
            ARGS="--seed $seed --lr $lr --data_dir $data_dir --domain_num $domain_num --max_epoch $max_epoch \
            --net $net --output $output --test_envs $test_envs --dataset $dataset --algorithm $algo \
            --gpu_id $gpu_id --loader_name daml_loader --is_data_aug yes \
            --is_different_class_space $c_space \
            $flag $val"

            python train.py $ARGS
            python eval.py $ARGS
            python eval_hscore.py $ARGS
            
        done
    done
done