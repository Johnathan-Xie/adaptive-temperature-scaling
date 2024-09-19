import os
import json
from typing import Literal, Optional
from dataclasses import dataclass, field

from datasets import DownloadMode


@dataclass
class EvaluationArguments:
    r"""
    Arguments pertaining to specify the evaluation parameters.
    """
    task: str = field(
        default=None,
        metadata={"help": "Name of the evaluation task."}
    )
    task_dir: Optional[str] = field(
        default="evaluation",
        metadata={"help": "Path to the folder containing the evaluation datasets."}
    )
    task_type: Optional[str] = field(
        default="mcq",
        metadata={"help": "mcq for multiple choice, frq for free response"}
    )
    batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "The batch size per GPU for evaluation."}
    )
    seed: Optional[int] = field(
        default=101,
        metadata={"help": "Random seed to be used with data loaders."}
    )
    lang: Optional[Literal["en", "zh"]] = field(
        default="en",
        metadata={"help": "Language used at evaluation."}
    )
    formatting: Optional[str] = field(
        default="en_mcq",
        metadata={"help": "Language used at evaluation."}
    )
    n_shot: Optional[int] = field(
        default=5,
        metadata={"help": "Number of examplars for few-shot learning."}
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the evaluation results."}
    )
    overwrite_save_dir: Optional[bool] = field(
        default=False,
    )
    download_mode: Optional[DownloadMode] = field(
        default=DownloadMode.REUSE_DATASET_IF_EXISTS,
        metadata={"help": "Download mode used for the evaluation datasets."}
    )
    confidence_method: Optional[Literal["logits", "tf"]] = field(
        default="logits",
        metadata={"help": "Method to calculate confidence. Logits for just softmax values and tf for true false evaluation"}
    )
    prechoice_softmax: Optional[bool] = field(
        default=False,
        #metadata={"when true, softmax then only select choices, when false only select choices then softmax"}
    )
    correctness_fn: Optional[str] = field(
        default="mcq"
    )
    stop_generation_regex: Optional[str] = field(
        default="\n"
    )
    tf_prompting_config: Optional[str] = field(
        default="post_rlhf"
    )
    temperature_scalar: Optional[float] = field(
        default=1.0
    )
    logit_reduction: Optional[str] = field(
        default="softmax_mean"
    )
    save_all_logits: Optional[bool] = field(
        default=False
    )
    scaling_binning_path: Optional[str] = field(
        default=None
    )
    def __post_init__(self):
        task_available = []
        for folder in os.listdir(self.task_dir):
            if os.path.isdir(os.path.join(self.task_dir, folder)):
                task_available.append(folder)
        with open(os.path.join(self.task_dir, "single_dataset_mapping.json")) as f:
            single_dataset_tasks = json.load(f)
        task_available = task_available + list(single_dataset_tasks.keys())
        if self.task not in task_available:
            raise ValueError("Task {} not found in {}.".format(self.task, self.task_dir))

        if self.save_dir is not None and os.path.exists(self.save_dir) and not self.overwrite_save_dir:
            raise ValueError("`save_dir` already exists, use another one.")
