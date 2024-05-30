#!/bin/bash
# ./scripts/run.sh "Dlinear" "No RevIN SAN DishTS" "ExchangeRate" "24"  "cuda:0" 336


# ./scripts/run.sh "Koopa" "No RevIN SAN DishTS" "ExchangeRate " "24"  "cuda:0" 336
# `./scripts/run.sh "Dlinear" "FANCP" "ExchangeRate " "24"  "cuda:0" 336`



# CUDA_DEVICE_ORDER=PCI_BUS_ID proxychains4 python3 ./torch_timeseries/norm_experiments/$model.py   --dataset_type="$dataset" --norm_type="$norm" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100 config_wandb --project=Norm --name="Norm"  runs --seeds='[1,2,3,4,5]'

declare -A dataset_to_window_map

# dataset_to_window_map["ETTm1"]=96
# dataset_to_window_map["ETTm2"]=96
# dataset_to_window_map["ETTh1"]=96
# dataset_to_window_map["ETTh2"]=96
# dataset_to_window_map["ExchangeRate"]=96
# dataset_to_window_map["ILI"]=36
# dataset_to_window_map["SP500"]=36
# dataset_to_window_map["Traffic"]=96
# dataset_to_window_map["Electricity"]=96
# dataset_to_window_map["Weather"]=96
# dataset_to_window_map["SolarEnergy"]=96

models=($1)
norms=($2)  
datasets=($3)  
pred_lens=($4)      
device=$5
windows=$6    
# ./run.sh "dlinear" "No" "ExchangeRate" "24"  "cuda:0"
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
                # CUDA_DEVICE_ORDER=PCI_BUS_ID HTTPS_PROXY="http://192.168.5.250:7890" https_proxy="http://192.168.5.250:7890" python3 ./torch_timeseries/norm_experiments/$model.py   --dataset_type="$dataset" --norm_type="$norm" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100 config_wandb --project=Norm --name="Norm"  runs --seeds='[1,2,3,4,5]'
                # CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/norm_experiments/$model.py   --dataset_type="$dataset" --norm_type="$norm" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100   runs --seeds='[1,2,3,4,5]'
                CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/norm_experiments/$model.py   --dataset_type="$dataset" --norm_type="$norm" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100   runs --seeds='[1]'
                # CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./torch_timeseries/norm_experiments/$model.py   --dataset_type="$dataset" --norm_type="$norm" --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100 config_wandb --project=Norm --name="Norm"  runs --seeds='[1,2,3,4,5]'
            done
        done
    done
done
echo "All runs completed."
