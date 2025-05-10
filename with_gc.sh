export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false

# GPU_IDS="0"
GPU_IDS="2"
# GPU_IDS="0,1,2,3,4,5,6,7"

# Training Configurations
learning_rate="1e-4"
#lr_schedule="cosine_with_restarts"
lr_schedule="constant"
optimizer="adamw"
steps="400"

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_1.yaml"
# ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_2.yaml"
# ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_8.yaml"

# Absolute path to where the data is located.
DATA_ROOT="video_dataset_camco/bike-packing_0_48_1"
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"
MASK_COLUMN="masks.txt"

# Launch experiments with different hyperparameters
#output_dir="results/cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"
output_dir="results_camco/bike-packing_0_48_1"

cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/cogvideox/cogvideox_image_to_video_lora_camco.py \
  --pretrained_model_name_or_path /workspace/hyeonho/CogVideoX-5b-I2V \
  --data_root $DATA_ROOT \
  --caption_column $CAPTION_COLUMN \
  --video_column $VIDEO_COLUMN \
  --mask_column $MASK_COLUMN \
  --height_buckets 480 \
  --width_buckets 720 \
  --frame_buckets 25 \
  --dataloader_num_workers 8 \
  --pin_memory \
  --validation_prompt \"a man is packing his bike in the house\" \
  --validation_images_dir \"inpainted\"
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_dir . \
  --validation_steps 50 \
  --validation_after_step 149 \
  --seed 42 \
  --rank 128 \
  --lora_alpha 128 \
  --mixed_precision bf16 \
  --output_dir $output_dir \
  --max_num_frames 25 \
  --train_batch_size 1 \
  --max_train_steps $steps \
  --checkpointing_steps 400 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --learning_rate $learning_rate \
  --lr_scheduler $lr_schedule \
  --lr_warmup_steps 0 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --noised_image_dropout 0.05 \
  --optimizer $optimizer \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 0.001 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb \
  --nccl_timeout 10000"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"
