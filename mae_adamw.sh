#!/bin/bash
#SBATCH -J mae
#SBATCH -p # partition name
#SBATCH -t # total running time
#SBATCH --nodes= # number of nodes to parallel
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/train_adamw.o%j  # Output file
#SBATCH -e logs/train_adamw.e%j   # Error file

source ~/.bashrc
conda deactivate
cd /path/to/softsign-mae
conda activate maeenv

MASTER_ADDR=$(srun --nodes=1 --ntasks=1 hostname -I | awk '{print $1}')
export MASTER_PORT=23456
export NODE_RANK=$SLURM_NODEID
export GPUS_PER_NODE= # number of GPUs per node

# export TMPDIR=/tmp
# export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache_${SLURM_PROCID}

# Downloading imagenet from huggingface requires authentication
# huggingface-cli login

srun python -u -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run_mae.py \
    --dataset_name ILSVRC/imagenet-1k \ #
    --output_dir vit-mae-adamw \
    --cache_dir /scratch/10152/baiyusu/mae/hf_cache \
    --remove_unused_columns False \
    --label_names pixel_values \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --overwrite_output_dir True \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --num_train_epochs 100 \ # train for 100 epochs
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 512 \ # takes ~60% of memory on a H100 80 GB GPU.
    --per_device_eval_batch_size 16 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --report_to wandb \ # upload training curves to wandb
    --run_name adamw-1.5e-4 \ # set run name on wandb
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --custom_optim adamw \ # choose adamw or softsign
    --trust_remote_code \
    --gradient_accumulation_steps 4 # make num_nodes * devices_per_node * per_device_bs * accumulation_steps = 4096

