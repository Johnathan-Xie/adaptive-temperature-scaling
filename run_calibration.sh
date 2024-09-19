main_process_port=25903

model="Qwen/Qwen-7B-Chat"
dataset="alpaca_gpt4_en"
calibration_config_path="calibration_configs/transformer_hidden_states"
label_smoothing=1.0
smooth_loss_weight=0.5
loss_type=selective_smoothing
label_smoothing_type=uniform
export RUN_NAME="calibration-$(basename $calibration_config_path)-loss_type_$loss_type-label_smoothing_type_$label_smoothing_type-smooth_loss_weight_$smooth_loss_weight-ls$label_smoothing-$(basename $model)-$dataset"
accelerate launch --config_file ddp_config.yaml --num_processes 4 --main_process_port $main_process_port \
    src/train_bash.py \
    --stage calibration \
    --model_name_or_path $model \
    --do_train \
    --dataset $dataset \
    --template default \
    --finetuning_type full \
    --output_dir runs/$RUN_NAME \
    --run_name $RUN_NAME \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --fp16 \
    --overwrite_output_dir \
    --seed 101 \
    --ddp_timeout 3000 \
    --calibration_config_path $calibration_config_path \
    --label_smoothing $label_smoothing \
    --smooth_loss_weight $smooth_loss_weight \
    --label_smoothing_type $label_smoothing_type \
    --loss_type $loss_type \
    --log_auxiliary_info \
    --save_total_limit 1 \