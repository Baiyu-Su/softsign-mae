#!/bin/bash
#SBATCH -J mae
#SBATCH -p gpu-h100
#SBATCH -t 48:00:00
#SBATCH --nodes=1                # Number of nodes to parallel
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/train_adamw.o%j  # Output file
#SBATCH -e logs/train_adamw.e%j   # Error file

source ~/.bashrc
conda deactivate
cd /scratch/10152/baiyusu/mae
conda activate maeenv


echo "Checking GPUs on each node..."
srun -l bash -c 'hostname; nvidia-smi -L'

cp -r /scratch/10152/baiyusu/mae/hf_cache /tmp/
# ls /scratch/10152/baiyusu/mae/data/imagenet2012/train

# srun --nodes=1 --ntasks=1 python test_compare.py

torchrun --standalone --nproc_per_node 2 run_mae.py \
    --dataset_name ILSVRC/imagenet-1k \
    --output_dir vit-mae-adamw \
    --cache_dir /tmp/hf_cache \
    --remove_unused_columns False \
    --label_names pixel_values \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --overwrite_output_dir True \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --num_train_epochs 50 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 512 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --report_to wandb \
    --run_name adamw-1.5e-4 \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --custom_optim adamw \
    --trust_remote_code \
    --gradient_accumulation_steps 4

