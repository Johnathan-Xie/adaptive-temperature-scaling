# Adaptive Temperature Scaling
Code for the paper ["Calibrating Language Models with Adaptive Temperature Scaling"]()

## Installation
```
conda create --name ats_env python=3.10
pip install -r requirements.txt
```
## Experiments
We provide 2 scripts to reproduce experiments from our paper.
run_calibration.sh will train a calibration head (with the settings from our method in the paper) with the specified base model.
```
bash run_calibration.sh
```
run_evaluation.sh will download and run evaluation for the specified model (either from the provided checkpoints on the hub or for your own weights).
```
bash run_evaluation.sh
```

The evaluation for truthful_qa will require an openai API key (for gpt3.5 evaluation).
While the code has been written to work with any language model from huggingface hub, we have only tested it with the three models used in our paper.
Additionally, the calibration code supports both DDP and FSDP, however we strongly recommend using the DDP config (already set by default) as
the FSDP training will require conversion of model checkpoints after training. Additionally, the model checkpoints produced are significantly larger.

## Code
The main code for calibration is in src/llmtuner/model/calibration. We write our implementation similar to huggingface peft modules. The calibration
model class wraps any huggingface causal language model and operates on top of the output logits. It applies the calibration head then the
calibration loss is used to fit the calibration head. When saving, only the calibration head is saved.

The main evaluation code is in src/llmtuner/eval/calibration_evaluator.py. Note that this evaluator does not support model sharding currently
meaning any model you evaluate must fit on a single GPU.

## Acknowledgement
This repo is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
