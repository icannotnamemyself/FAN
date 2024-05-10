#!/bin/bash
# ./norm_script/run.sh "dlinear" "No RevIN SAN DishTS" "ExchangeRate" "24"  "cuda:0"

declare -A dataset_to_window_map

dataset_to_window_map["ETTm1"]=96
dataset_to_window_map["ETTm2"]=96
dataset_to_window_map["ETTh1"]=96
dataset_to_window_map["ETTh2"]=96
dataset_to_window_map["ExchangeRate"]=96
dataset_to_window_map["ILI"]=36
dataset_to_window_map["SP500"]=36
dataset_to_window_map["Traffic"]=96
dataset_to_window_map["Electricity"]=96
dataset_to_window_map["Weather"]=96
dataset_to_window_map["SolarEnergy"]=96

model=$1
norms=($2)  
datasets=($3)  
pred_lens=($4)      
device=$5
# windows=$4        
# ./run.sh "dlinear" "No" "ExchangeRate" "24"  "cuda:0"

for norm in "${norms[@]}"
do
    for dataset in "${datasets[@]}"
    do
        windows=${dataset_to_window_map[$dataset]}
        for pred_len in "${pred_lens[@]}"
        do
            echo "Running with dataset = $dataset and pred_len = $pred_len , window = $windows"
            CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/norm_experiments/$model.py   --dataset_type="$dataset" --norm_type="$norm" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100 config_wandb --project=Norm --name="Norm"  runs --seeds='[1,2]'
        done
    done
done

echo "All runs completed."