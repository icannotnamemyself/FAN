#!/bin/bash

declare -A dataset_to_window_map


models=($1)
norms=($2)  
datasets=($3)  
pred_lens=($4)      
device=$5
windows=$6
norm_config=$7
scaler_type=$8
for model in "${models[@]}"
    do
    for norm in "${norms[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for pred_len in "${pred_lens[@]}"
            do
                CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/norm_experiments/$model.py --dataset_type="$dataset" --norm_type="$norm" --norm_config=$norm_config --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --scaler_type="$scaler_type" --epochs=100 config_wandb --project=Norm --name="Norm"  runs --seeds='[1,2,3,4,5]'
            done
        done
    done
done

echo "All runs completed."
