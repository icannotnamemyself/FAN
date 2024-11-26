export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./torch_timeseries/norm_experiments/DLinear.py \
   --dataset_type="ETTh1" \
   --split_type="popular" \
   --norm_type="FAN" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len="96" \
   --windows=96 \
   --epochs=100   \
   --norm_config="{freq_topk:4}" \
   runs --seeds='[3]'
