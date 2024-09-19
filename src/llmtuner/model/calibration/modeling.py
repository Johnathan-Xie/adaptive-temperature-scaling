import collections
import inspect
import os

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn

from transformers import AutoModelForCausalLM
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory, is_npu_available, is_xpu_available
from safetensors.torch import load_file as safe_load_file

from huggingface_hub import hf_hub_download, file_exists
from huggingface_hub.utils import EntryNotFoundError

from safetensors.torch import save_file as safe_save_file
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.utils import PushToHubMixin
from peft.utils import id_tensor_storage

from .config import CalibrationConfig
from .head import ARCHITECTURE_TYPE_TO_HEAD_MAPPING

CALIBRATION_WEIGHTS_NAME = "calibration_head.bin"
CALIBRATION_SAFETENSORS_WEIGHTS_NAME = "calibration_head.safetensors"

CLASS_MAPPING = {}

def infer_device():
    if torch.cuda.is_available():
        torch_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    elif is_xpu_available():
        torch_device = "xpu"
    elif is_npu_available():
        torch_device = "npu"
    else:
        torch_device = "cpu"
    return torch_device

def load_calibration_head_weights(model_id: str, device: Optional[str] = None, **hf_hub_download_kwargs) -> dict:
    path = (
        os.path.join(model_id, hf_hub_download_kwargs["subfolder"])
        if hf_hub_download_kwargs.get("subfolder", None) is not None
        else model_id
    )

    if device is None:
        device = infer_device()
    
    if os.path.exists(os.path.join(path, CALIBRATION_SAFETENSORS_WEIGHTS_NAME)):
        filename = os.path.join(path, CALIBRATION_SAFETENSORS_WEIGHTS_NAME)
        use_safetensors = True
    elif os.path.exists(os.path.join(path, CALIBRATION_WEIGHTS_NAME)):
        filename = os.path.join(path, CALIBRATION_WEIGHTS_NAME)
        use_safetensors = False
    else:
        token = hf_hub_download_kwargs.get("token", None)
        if token is None:
            token = hf_hub_download_kwargs.get("use_auth_token", None)

        has_remote_safetensors_file = file_exists(
            repo_id=model_id,
            filename=CALIBRATION_SAFETENSORS_WEIGHTS_NAME,
            revision=hf_hub_download_kwargs.get("revision", None),
            repo_type=hf_hub_download_kwargs.get("repo_type", None),
            token=token,
        )
        use_safetensors = has_remote_safetensors_file

        if has_remote_safetensors_file:
            # Priority 1: load safetensors weights
            filename = hf_hub_download(
                model_id,
                CALIBRATION_SAFETENSORS_WEIGHTS_NAME,
                **hf_hub_download_kwargs,
            )
        else:
            try:
                filename = hf_hub_download(model_id, CALIBRATION_WEIGHTS_NAME, **hf_hub_download_kwargs)
            except EntryNotFoundError:
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {CALIBRATION_WEIGHTS_NAME} or {CALIBRATION_SAFETENSORS_WEIGHTS_NAME} is present at {model_id}."
                )

    if use_safetensors:
        if hasattr(torch.backends, "mps") and (device == torch.device("mps")):
            confidence_head_weights = safe_load_file(filename, device="cpu")
        else:
            confidence_head_weights = safe_load_file(filename, device=device)
    else:
        confidence_head_weights = torch.load(filename, map_location=torch.device(device))

    return confidence_head_weights

class CalibrationModel(PushToHubMixin, nn.Module):
    """
    Base model encompassing various Peft methods.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.

    **Attributes**:
        - **base_model** ([`torch.nn.Module`]) -- The base transformer model used for Peft.
        - **platt_config** ([`PeftConfig`]) -- The configuration of the Platt model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
            saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
            using [`PromptLearningConfig`].
        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Peft if
            using [`PromptLearningConfig`].
        - **transformer_backbone_name** (`str`) -- The name of the transformer
            backbone in the base model if using [`PromptLearningConfig`].
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
            in the base model if using [`PromptLearningConfig`].
    """

    def __init__(self, model: PreTrainedModel, calibration_config: CalibrationConfig) -> None:
        super().__init__()
        self.calibration_type = calibration_config.calibration_type

        self.calibration_config = calibration_config
        calibration_cls = ARCHITECTURE_TYPE_TO_HEAD_MAPPING[calibration_config.calibration_type]
        self.calibration_head = calibration_cls(**calibration_config.to_dict(), base_model=model)
        
        #if getattr(model, "is_gradient_checkpointing", True):
        #    model = self._prepare_model_for_gradient_checkpointing(model)
        self.base_model = model
        self.overwrite_logits = False
        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        #if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
        #    self.base_model.config.pretraining_tp = 1
        if calibration_config.freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
    def set_overwrite_logits(self, set_to=True):
        self.overwrite_logits = set_to

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        is_main_process: bool = True,
        **kwargs: Any,
    ) -> None:
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            #self.create_or_update_model_card(save_directory)

        calibration_config = self.calibration_config
        # save only the trainable weights
        output_state_dict = {k:v for k,v in self.calibration_head.state_dict().items()}
        
        output_dir = save_directory
        os.makedirs(output_dir, exist_ok=True)

        if is_main_process and safe_serialization:
            # Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs = collections.defaultdict(list)
            for name, tensor in output_state_dict.items():
                # Sometimes in the state_dict we have non-tensor objects.
                # e.g. in bitsandbytes we have some `str` objects in the state_dict
                if isinstance(tensor, torch.Tensor):
                    ptrs[id_tensor_storage(tensor)].append(name)
                else:
                    # In the non-tensor case, fall back to the pointer of the object itself
                    ptrs[id(tensor)].append(name)

            # These are all the pointers of shared tensors.
            shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

            for _, names in shared_ptrs.items():
                # Here we just clone the shared tensors to avoid tensor aliasing which is
                # not supported in safetensors.
                for shared_tensor_name in names[1:]:
                    output_state_dict[shared_tensor_name] = output_state_dict[shared_tensor_name].clone()

            safe_save_file(
                output_state_dict,
                os.path.join(output_dir, CALIBRATION_SAFETENSORS_WEIGHTS_NAME),
                metadata={"format": "pt"},
            )
        elif is_main_process:
            torch.save(output_state_dict, os.path.join(output_dir, CALIBRATION_WEIGHTS_NAME))
        
        inferred_base_model_name = self.base_model.__dict__.get("name_or_path", None)
        if inferred_base_model_name is not None:
            calibration_config.base_model_name_or_path = inferred_base_model_name
                        
        if is_main_process:
            calibration_config.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, os.PathLike],
        base_model: torch.nn.Module = None,
        is_trainable: bool = False,
        config: Optional[CalibrationConfig] = None,
        **kwargs: Any,
    ) -> "CalibrationModel":
        # load the config
        if config is None:
            config = CalibrationConfig.from_pretrained(model_id, **kwargs)
        elif isinstance(config, CalibrationConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a CalibrationConfig, got {config.__class__}")
        if base_model is None:
            #TODO: NEEDS FIX TO ADAPT. NOT REALLY NECESSARY, BUT GOOD FOR "CORRECTNESS"
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                config=config,
                **kwargs
            )
        else:
            # Correct base model path in config, Probably not good to try and access pseudo private attributes tho
            config.base_model_name_or_path = base_model._name_or_path
        if config.task_type not in CALIBRATION_TYPE_TO_MODEL_MAPPING.keys():
            model = cls(base_model, config)
        else:
            model = CALIBRATION_TYPE_TO_MODEL_MAPPING[config.task_type](base_model, config)
        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)
        
        model.load_calibration_head(model_id, is_trainable=is_trainable, **kwargs)
        return model
    
    def load_calibration_head(self, model_id: str, is_trainable: bool = False, **kwargs: Any):
        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        torch_device = infer_device()
        
        confidence_head_weights = load_calibration_head_weights(model_id, device=torch_device, **hf_hub_download_kwargs)

        # load the weights into the model
        load_result = self.calibration_head.load_state_dict(confidence_head_weights, strict=False)
        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            offload_dir = kwargs.get("offload_folder", None)
            offload_index = kwargs.get("offload_index", None)

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = self._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )
            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )
            hook = AlignDevicesHook(io_same_device=True)
            add_hook_to_module(self.base_model, hook)

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result

    @classmethod
    def _split_kwargs(cls, kwargs: Dict[str, Any]):
        _kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
        hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters or key in _kwargs_not_in_hf_hub_download_signature:
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, other_kwargs
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        if kwargs.get("hidden_states") is not None:
            outputs = CausalLMOutput(logits=kwargs.get("logits"), hidden_states=[hidden_states])
            # We probably won't cache logits since there are so many
            if outputs.logits is None:
                logits = self.base_model.lm_head(outputs.hidden_states[-1])
        else:
            outputs = self.base_model(*args, **kwargs)
        outputs = self.calibration_head(outputs, attention_mask=kwargs.get("attention_mask"))
        if self.overwrite_logits:
            outputs.logits = outputs.calibrated_logits
        return outputs

class TopKSmoothingLoss(nn.Module):
    def __init__(self, k=5, label_smoothing=0.1, reduction="mean"):
        super().__init__()
        self.k = k
        self.crossentropy = CrossEntropyLoss(reduction=reduction)
        self.uniform_weight = label_smoothing
        self.hard_weight = 1 - label_smoothing
        self.auxiliary_logs = {}
        
    def forward(self, logits, labels, dim=-1):
        topk_indices = logits.topk(k=self.k, dim=dim).indices
        
        uniform_labels = torch.zeros_like(logits)
        uniform_labels = uniform_labels.scatter_(index=topk_indices, src=torch.ones_like(topk_indices) / self.k, dim=dim)
        
        uniform_loss = self.crossentropy(logits, uniform_labels) * self.uniform_weight
        hard_loss = self.crossentropy(logits, labels) * self.hard_weight
        loss = uniform_loss + hard_loss
        
        return loss
    
class SelectiveSmoothingLoss(nn.Module):
    def __init__(
        self,
        label_smoothing_type="topk",
        smooth_loss_weight=0.5,
        label_smoothing=0.5,
        weighted_average=True,
        k=5,
        threshold=0.1,
        **kwargs
    ):
        super().__init__()
        """weighted_average means the loss terms will be weighted based on how many samples belong to each term"""
        self.hard_loss = nn.CrossEntropyLoss(reduction="none")
        self.weighted_average = weighted_average
        self.smooth_loss_weight = smooth_loss_weight
        if label_smoothing_type == "uniform":
            self.smooth_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")
        elif label_smoothing_type == "topk":
            self.smooth_loss = TopKSmoothingLoss(label_smoothing=label_smoothing, k=k, reduction="none")
        elif label_smoothing_type == "threshold":
            self.smooth_loss = ThresholdSmoothingLoss(label_smoothing=label_smoothing, threshold=threshold, reduction="none")
        else:
            raise ValueError(f"Unknown label_smoothing_type: {label_smoothing_type}")
    
    def forward(self, logits, labels, dim=-1):
        correct_mask = logits.argmax(dim=dim) == labels
        
        hard_loss = self.hard_loss(logits[correct_mask], labels[correct_mask]) if correct_mask.sum() > 0 else torch.Tensor([0])
        smooth_loss = self.smooth_loss(logits[~correct_mask], labels[~correct_mask]) if (~correct_mask).sum() > 0 else torch.Tensor([0])

        if self.weighted_average:
            smooth_weight = self.smooth_loss_weight * (correct_mask.sum() / correct_mask.numel())
            hard_weight = (1 - self.smooth_loss_weight) * ((~correct_mask).sum() / correct_mask.numel())
            total_weight = smooth_weight + hard_weight
            smooth_weight = (1 / total_weight) * smooth_weight
            hard_weight = (1 / total_weight) * hard_weight
            hard_loss = (hard_loss * hard_weight.to(hard_loss.device)).mean()
            smooth_loss = (smooth_loss * smooth_weight.to(smooth_loss.device)).mean()
        else:
            smooth_weight = self.smooth_loss_weight
            hard_weight = (1 - self.smooth_loss_weight)
            hard_loss = (hard_loss.mean() * hard_weight.to(hard_loss.device))
            smooth_loss = (smooth_loss.mean() * smooth_weight.to(smooth_loss.device))
        return hard_loss + smooth_loss
    
def get_lm_loss_fn(calibration_config):
    if calibration_config.loss_type == "xent":
        return CrossEntropyLoss(label_smoothing=calibration_config.label_smoothing)
    elif calibration_config.loss_type == "topk_smoothing":
        return TopKSmoothingLoss(
            label_smoothing=calibration_config.label_smoothing,
            k=calibration_config.smoothing_topk,
        )
    elif calibration_config.loss_type == "selective_smoothing":
        return SelectiveSmoothingLoss(
            label_smoothing=calibration_config.label_smoothing,
            label_smoothing_type=calibration_config.label_smoothing_type,
            smooth_loss_weight=calibration_config.smooth_loss_weight,
            k=calibration_config.smoothing_topk,
        )
    else:
        raise ValueError(f"Unknown loss type: {calibration_config.loss_type}")

class CalibrationModelForCausalLM(CalibrationModel):
    """
    Peft model for causal language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.


    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModelForCausalLM, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "CAUSAL_LM",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 1280,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 20,
        ...     "num_layers": 36,
        ...     "encoder_hidden_size": 1280,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        >>> peft_model = PeftModelForCausalLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
        ```
    """

    def __init__(self, model: torch.nn.Module, calibration_config: CalibrationConfig) -> None:
        super().__init__(model, calibration_config)
        self.prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.loss_fn = get_lm_loss_fn(calibration_config)
    
    def forward(
        self,
        *args,
        **kwargs,
    ):
        outputs = super().forward(*args, **kwargs, output_hidden_states=True)
        #outputs = self.base_model(*args, **kwargs, output_hidden_states=True)
        #outputs = self.calibration_head(outputs)
        if kwargs.get("labels") is not None:
            labels = kwargs["labels"]
            logits = outputs.calibrated_logits
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fn(shift_logits, shift_labels)
            outputs.loss = loss
        return outputs

CALIBRATION_TYPE_TO_MODEL_MAPPING: Dict[str, CalibrationModel] = {
    "CAUSAL_LM": CalibrationModelForCausalLM,
}