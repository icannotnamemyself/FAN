#!/bin/bash





declare -A dataset_to_window_map


models=($1)
norms=($2)  
datasets=($3)  
pred_lens=($4)      
device=$5
windows=$6    
for model in "${models[@]}"
do
    for norm in "${norms[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            # windows=${dataset_to_window_map[$dataset]}
            for pred_len in "${pred_lens[@]}"
            do
                echo "Running with dataset = $dataset and pred_len = $pred_len , window = $windows"
                CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/norm_experiments/$model.py   --dataset_type="$dataset" --norm_type="$norm" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100  config_wandb --project=Norm --name="Norm"  runs --seeds='[1,2,3,4,5]'
            done
        done
    done
done
echo "All runs completed."
