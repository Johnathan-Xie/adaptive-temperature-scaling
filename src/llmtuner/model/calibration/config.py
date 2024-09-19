import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Union

from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin

CONFIG_NAME = "config.json"

@dataclass
class CalibrationConfigMixin(PushToHubMixin):
    r"""
    This is the base configuration class for PEFT adapter models. It contains all the methods that are common to all
    PEFT adapter models. This class inherits from [`~transformers.utils.PushToHubMixin`] which contains the methods to
    push your model to the Hub. The method `save_pretrained` will save the configuration of your adapter model in a
    directory. The method `from_pretrained` will load the configuration of your adapter model from a directory.

    Args:
        peft_type (Union[[`~peft.utils.config.PeftType`], `str`]): The type of Peft method to use.
    """
    calibration_type: Optional[str] = field(default="temperature", metadata={"help": "The type of PEFT model."})

    def to_dict(self) -> Dict:
        r"""
        Returns the configuration for your adapter model as a dictionary.
        """
        return asdict(self)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        r"""
        This method saves the configuration of your adapter model in a directory.

        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the [`~transformers.utils.PushToHubMixin.push_to_hub`]
                method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)
        auto_mapping_dict = kwargs.pop("auto_mapping_dict", None)

        output_dict = asdict(self)
        # converting set type to list
        for key, value in output_dict.items():
            if isinstance(value, set):
                output_dict[key] = list(value)

        output_path = os.path.join(save_directory, CONFIG_NAME)

        # Add auto mapping details for custom models.
        if auto_mapping_dict is not None:
            output_dict["auto_mapping"] = auto_mapping_dict

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: Optional[str] = None, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a directory.

        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the Hub repository id where the configuration is saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the child class initialization.
        """
        # Avoid circular dependency .. TODO: fix this with a larger refactor
        

        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    pretrained_model_name_or_path, CONFIG_NAME, subfolder=subfolder, **hf_hub_download_kwargs
                )
            except Exception:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)

        # TODO: this hack is needed to fix the following issue (on commit 702f937):
        # if someone saves a default config and loads it back with `PeftConfig` class it yields to
        # not loading the correct config class.

        # from peft import AdaLoraConfig, PeftConfig
        # peft_config = AdaLoraConfig()
        # print(peft_config)
        # >>> AdaLoraConfig(peft_type=<PeftType.ADALORA: 'ADALORA'>, auto_mapping=None, base_model_name_or_path=None,
        # revision=None, task_type=None, inference_mode=False, r=8, target_modules=None, lora_alpha=8, lora_dropout=0.0, ...
        #
        # peft_config.save_pretrained("./test_config")
        # peft_config = PeftConfig.from_pretrained("./test_config")
        # print(peft_config)
        # >>> PeftConfig(peft_type='ADALORA', auto_mapping=None, base_model_name_or_path=None, revision=None, task_type=None, inference_mode=False)

        config_cls = cls

        kwargs = {**class_kwargs, **loaded_attributes}
        config = config_cls(**kwargs)
        return config

    @classmethod
    def from_json_file(cls, path_json_file: str, **kwargs):
        r"""
        Loads a configuration file from a json file.

        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object

    @classmethod
    def _split_kwargs(cls, kwargs):
        hf_hub_download_kwargs = {}
        class_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value
            elif key in list(cls.__annotations__):
                class_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, class_kwargs, other_kwargs

    @classmethod
    def _get_peft_type(
        cls,
        model_id: str,
        **hf_hub_download_kwargs,
    ):
        subfolder = hf_hub_download_kwargs.get("subfolder", None)

        path = os.path.join(model_id, subfolder) if subfolder is not None else model_id

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    model_id,
                    CONFIG_NAME,
                    **hf_hub_download_kwargs,
                )
            except Exception:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{model_id}'")

        loaded_attributes = cls.from_json_file(config_file)
        return loaded_attributes["peft_type"]

    @property
    def is_prompt_learning(self) -> bool:
        r"""
        Utility method to check if the configuration is for prompt learning.
        """
        return False

    @property
    def is_adaption_prompt(self) -> bool:
        """Return True if this is an adaption prompt config."""
        return False


@dataclass
class CalibrationConfig(CalibrationConfigMixin):
    """
    This is the base configuration class to store the configuration of a [`PeftModel`].

    Args:
        peft_type (Union[[`~peft.utils.config.PeftType`], `str`]): The type of Peft method to use.
        task_type (Union[[`~peft.utils.config.TaskType`], `str`]): The type of task to perform.
        inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
    """

    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the base model to use."}
    )
    inference_mode: Optional[bool] = field(
        default=False,
    )
    freeze_base_model: Optional[bool] = field(
        default=True
    )
    calibration_type: Optional[str] = field(default="temperature", metadata={"help": "The type of calibration model."})
    init_temperature: Optional[float] = field(default=1.0)
    task_type: Optional[str] = field(default="CAUSAL_LM")
    in_features: Optional[int] = field(default=4096)
    intermediate_size: Optional[int] = field(default=11008)
    max_position_embeddings: Optional[int] = field(default=4096)
    layer_idx: Optional[int] = field(default=33)
    attention_dropout: Optional[float] = field(default=0.0)
    num_attention_heads: Optional[int] = field(default=32)
    num_key_value_heads: Optional[int] = field(default=32)
    hidden_act: Optional[str] = field(default="silu")
    feature_key: Optional[str] = field(default="hidden_states")
    normalize_logits: Optional[bool] = field(default=False)
    label_smoothing: Optional[float] = field(default=0.0)
    smooth_loss_weight: Optional[float] = field(default=0.5)
    smoothing_topk: Optional[int] = field(default=5)
    loss_type: Optional[str] = field(default="xent")
    label_smoothing_type: Optional[str] = field(default="topk")
    log_auxiliary_info: Optional[bool] = field(default=False)