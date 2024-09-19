# checkpoint path (one of)
# 1. jxie/ATS-calibration-Llama-2-7b-chat-hf-alpaca_gpt4_en
# 2. jxie/ATS-calibration-Llama-2-13b-chat-hf-alpaca_gpt4_en 
# 3. jxie/ATS-calibration-Qwen-7B-Chat-alpaca_gpt4_en
# model (one of)
# 1. meta-llama/Llama-2-7b-chat-hf
# 2. meta-llama/Llama-2-13b-chat-hf
# 3. Qwen/Qwen-7B-Chat
checkpoint_path=jxie/ATS-calibration-Llama-2-7b-chat-hf-alpaca_gpt4_en
model=meta-llama/Llama-2-7b-chat-hf
template="vanilla"

# MMLU Evaluation
correctness_fn="mcq"
task_type="mcq"
cm="logits"
formatting="en_mcq"
task="mmlu"
ps=false

CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
    --model_name_or_path $model \
    --template $template \
    --task $task \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 1 \
    --save_dir evaluation_results/$task/$(basename $checkpoint_path) \
    --overwrite_save_dir \
    --checkpoint_dir $checkpoint_path \
    --seed 101 \
    --finetuning_type full \
    --prechoice_softmax $ps \
    --confidence_method $cm \
    --correctness_fn $correctness_fn \
    --task_type $task_type \
    --formatting $formatting \
    --is_calibration_model


# TriviaQA Evaluation
correctness_fn="frq_em"
task_type="frq"
formatting="en_frq"
task=trivia_qa

CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
    --model_name_or_path $model \
    --template $template \
    --task $task \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 1 \
    --save_dir evaluation_results/$task/$(basename $checkpoint_path) \
    --overwrite_save_dir \
    --checkpoint_dir $checkpoint_path \
    --seed 101 \
    --finetuning_type full \
    --prechoice_softmax $ps \
    --confidence_method $cm \
    --correctness_fn $correctness_fn \
    --task_type $task_type \
    --formatting $formatting \
    --max_new_tokens 16 \
    --do_sample true \
    --use_calibrated_logits \
    --is_calibration_model

: << UNCOMMENT
# TruthfulQA Evaluation
# Note this evaluation requires a valid azure openai API Key. Set the $OPENAI_API_KEY envionment variable. Remove the comment to run this when set
correctness_fn="frq_lm_gpt35"
formatting="truthful_qa"
task="truthful_qa"
CUDA_VISIBLE_DEVICES=0 python src/evaluate.py \
    --model_name_or_path $model \
    --template $template \
    --task $task \
    --split test \
    --lang en \
    --n_shot 0 \
    --batch_size 1 \
    --save_dir evaluation_results/$task/$(basename $checkpoint_path) \
    --overwrite_save_dir \
    --checkpoint_dir $checkpoint_path \
    --seed 101 \
    --finetuning_type full \
    --prechoice_softmax $ps \
    --confidence_method $cm \
    --correctness_fn $correctness_fn \
    --task_type $task_type \
    --formatting $formatting \
    --max_new_tokens 64 \
    --do_sample true \
    --use_calibrated_logits \
    --is_calibration_model
UNCOMMENT