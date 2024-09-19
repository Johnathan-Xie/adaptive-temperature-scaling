# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py

import os
import json
import torch
import inspect
import tiktoken
import numpy as np
import math
from sklearn.metrics import roc_auc_score
import re
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from typing import Any, Dict, List, Optional
from collections.abc import Iterable

from datasets import load_dataset
from transformers.utils import cached_file
from transformers import GenerationConfig

from llmtuner.data.template import get_template_and_fix_tokenizer
from llmtuner.eval.template import get_eval_template
from llmtuner.eval.correctness import get_correctness_fn
from llmtuner.extras.constants import CHOICES
from llmtuner.model import dispatch_model, get_eval_args, load_model_and_tokenizer


def create_ece_plot(
    confidence_bin_values,
    accuracy_bin_values,
    bin_counts,
    output_file=None,
    color="red",
    exp_factor=1.0,
    ece_value=None,
):
    bin_counts = np.array(bin_counts)
    alpha_weights = np.array(bin_counts / bin_counts.sum())
    alpha_weights /= alpha_weights.max()
    alpha_weihgts = alpha_weights ** exp_factor
    for bin_idx in range(len(confidence_bin_values)):
        plt.bar(
            [(confidence_bin_values[bin_idx][1] + confidence_bin_values[bin_idx][0]) / 2],
            [accuracy_bin_values[bin_idx]],
            color=color,
            alpha=alpha_weights[bin_idx],
            width=confidence_bin_values[bin_idx][1] - confidence_bin_values[bin_idx][0],
        )
    # For perfect calibration line
    plt.plot((0, 1))
    plt.xlabel("Confidence Bin")
    plt.ylabel("Accuracy in Bin")
    if ece_value is not None:
        plt.text(0.1, 0.8, f"Calibration Error: {ece_value}")
    if output_file is not None:
        plt.savefig(output_file)
    plt.clf()

def compute_ece(corrects, confidences, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    confidence_bin_values, accuracy_bin_values, bin_counts = [], [], []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = (confidences > bin_lower) & (confidences < bin_upper.item())
        confidence_bin_values.append((bin_lower.item(), bin_upper.item()))
        proportion_in_bin = in_bin.mean()
        bin_counts.append(int(in_bin.sum()))
        if proportion_in_bin.item() > 0:
            accuracy_in_bin = corrects[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * proportion_in_bin
            accuracy_bin_values.append(accuracy_in_bin.item())
        else:
            accuracy_bin_values.append(math.nan)
    return ece, confidence_bin_values, accuracy_bin_values, bin_counts

def compute_brier_score(corrects, confidences):
    return ((confidences - corrects) ** 2).mean().item()

def compute_best_temperature(
    logits,
    corrects,
    num_values=1000,
    metric_fn=compute_ece,
    lower_is_better=True,
    selected_token_indices=None,
):
    temperatures = 10 ** torch.linspace(-1.25, 2, num_values)
    
    logits = torch.Tensor(logits).unsqueeze(0)
    corrects = np.array(corrects)
    temperatures = temperatures.unsqueeze(-1).unsqueeze(-1)
    logits = logits / temperatures
    if selected_token_indices is not None:
        probs = torch.nn.functional.softmax(logits, dim=-1)[:, selected_token_indices]
    else:
        probs = torch.nn.functional.softmax(logits, dim=-1)
    confidences = probs.max(dim=-1)[0].numpy()
    best_metric, best_temperature, extra_info = None, None, ()

    for idx, t in enumerate(temperatures):
        metric_outputs = metric_fn(corrects, confidences[idx])
        if isinstance(metric_outputs, Iterable):
            metric_value = metric_outputs[0]
        else:
            metric_value = metric_outputs
            metric_outputs = (metric_outputs, )
        if best_metric is None or (lower_is_better and metric_value < best_metric) or (not lower_is_better and metric_value > best_metric):
            best_metric = metric_value
            best_temperature = t
            extra_info = metric_outputs[1:]
        
    return best_metric, best_temperature.item(), *extra_info

def aggregate_metrics(
    metrics_dict,
    category_key=None,
    keys_to_aggregate=["corrects", "confidences", "predictions", "labels", "logits", "entropies"],
):
    "If aggregation key is None, then we will aggregate everything into a single group"
    aggregated_metrics = {}
    for metrics in metrics_dict.values():
        category = metrics[category_key] if category_key is not None else "Average"
        if category in aggregated_metrics:
            for metric_key, metric_value in metrics.items():
                if metric_key in keys_to_aggregate:
                    aggregated_metrics[category][metric_key].extend(metric_value)
        else:
            aggregated_metrics[category] = {}
            for metric_key, metric_value in metrics.items():
                if metric_key in keys_to_aggregate:
                    aggregated_metrics[category][metric_key] = metric_value
    return aggregated_metrics

def add_metrics(metrics_dict):
    confidences = np.array(metrics_dict["confidences"])
    corrects = np.array(metrics_dict["corrects"])
    entropies = np.array(metrics_dict["entropies"])

    new_metrics = {}
    new_metrics["auc"] = roc_auc_score(corrects, confidences)
    if len(metrics_dict["entropies"]) == len(metrics_dict["corrects"]):
        new_metrics["auc_entropy"] = roc_auc_score(corrects, entropies)
    
    (
        new_metrics["ece"], 
        new_metrics["confidence_bin_values"], 
        new_metrics["accuracy_bin_values"], 
        new_metrics["bin_counts"]
    ) = compute_ece(corrects, confidences)
    
    new_metrics["brier_score"] = compute_brier_score(corrects, confidences)


    if len(metrics_dict["logits"]) == len(metrics_dict["corrects"]):
        (
            new_metrics["best_ece"], 
            new_metrics["best_ece_temperature"],
            new_metrics["best_confidence_bin_values"], 
            new_metrics["best_accuracy_bin_values"], 
            new_metrics["best_bin_counts"], 
        ) = compute_best_temperature(metrics_dict["logits"], metrics_dict["corrects"], metric_fn=compute_ece)
        new_metrics["best_brier_score"], new_metrics["best_brier_score_temperature"] = compute_best_temperature(metrics_dict["logits"], metrics_dict["corrects"], metric_fn=compute_brier_score)

    new_metrics["accuracy"] = 100 * corrects.mean().item()
    metrics_dict.update(new_metrics)
    return metrics_dict

def format_score_info(metrics):
    score_info = ""
    for category in metrics.keys():
        score_info += f"{category}\n"
        for metrics_key, metrics_value in metrics[category].items():
            if isinstance(metrics[category][metrics_key], list):
                continue
            score_info += f"    {metrics_key}: {metrics_value}\n"
    return score_info


class CalibrationEvaluator:

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args, self.generating_args = get_eval_args(args)
        self.model_args.compute_dtype = torch.float32
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_args, finetuning_args)
        if self.model_args.use_calibrated_logits and hasattr(self.model, ("set_overwrite_logits")):
            self.model.set_overwrite_logits(True)
        
        generation_args_dict = {k: v for k, v in self.generating_args.to_dict().items() if v is not None}
        self.generation_config = GenerationConfig.from_pretrained(self.model_args.model_name_or_path)
        self.generation_kwargs = json.loads(self.generation_config.to_json_string(ignore_metadata=True))
        # Fix for Qwen
        self.generation_kwargs.pop("chat_format", None)
        self.generation_kwargs.pop("max_window_size", None)
        self.generation_kwargs = self.generation_kwargs | generation_args_dict

        if self.generation_kwargs.get("max_new_tokens", -1) > 0:
            self.generation_kwargs.pop("max_length", None)
        else:
            self.generation_kwargs.pop("max_new_tokens", None)
        
        self.tokenizer.padding_side = "right" if "mcq" in self.eval_args.formatting else "left"
        self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer(self.data_args.template, self.tokenizer)
        self.eval_template = get_eval_template(self.eval_args.formatting)
        self.choice_inputs = self._encode_choices() 
        self.correctness_fn = get_correctness_fn(self.eval_args.correctness_fn)

    def _encode_choices(self) -> List[int]:
        if isinstance(getattr(self.tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        return [self.tokenizer.encode(self.eval_template.prefix + ch, **kwargs)[-1] for ch in CHOICES]

    @torch.inference_mode()
    def batch_inference_mcq(self, batch_input: Dict[str, torch.Tensor]) -> List[str]:
        model_outputs = self.model(**batch_input)
        logits = model_outputs.calibrated_logits if hasattr(model_outputs, "calibrated_logits") else model_outputs.logits
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        all_token_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0).cpu()
        choice_probs = self.get_softmax_probs(all_token_probs, self.choice_inputs)
        
        probs, offsets = torch.max(choice_probs, dim=-1)
        logits_to_save = (all_token_probs).detach().tolist()
        return [chr(ord("A") + offset.item()) for offset in offsets], [i.item() for i in probs.cpu()], logits_to_save
    
    @torch.no_grad()
    def reduce_logits(self, logits, positions):
        if self.eval_args.logit_reduction == "softmax_mean":
            probs = torch.nn.functional.softmax(logits / self.eval_args.temperature_scalar, dim=-1)
            selected_probs = probs[torch.arange(logits.shape[0]).long(), positions]
            return selected_probs.mean().cpu().item() if len(selected_probs) else 0.0

    @torch.no_grad()
    def batch_inference_frq(self, batch_input: Dict[str, torch.Tensor], stop_generation_token="\n", **generation_kwargs) -> List[str]:
        batch_size = len(batch_input["input_ids"])
        generated_sequences = self.model.generate(**batch_input, **generation_kwargs)
        
        all_predicted = []
        for i in range(batch_size):
            tokenized_question_len = len(batch_input["input_ids"][i])
            tokenized_prediction = generated_sequences[i]
            tokenized_prediction = tokenized_prediction[tokenized_question_len:].cpu().tolist()
            string_prediction = self.tokenizer.decode(tokenized_prediction, skip_special_tokens=True)

            string_prediction = string_prediction.lstrip()
            stop_indices = [int(m.start(0)) for m in re.finditer(stop_generation_token, string_prediction)]
            stop_idx = stop_indices[0] if len(stop_indices) > 0 else len(string_prediction)
            string_prediction = string_prediction[:stop_idx]
            all_predicted.append(string_prediction)
        
        all_confidences = []
        if self.eval_args.confidence_method == "logits":
            question_strings = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch_input["input_ids"]]
            tokenized_answers = [self.tokenizer(all_predicted[i])["input_ids"][1:] for i in range(batch_size)]
            full_input_ids = [self.tokenizer(question_strings[i])["input_ids"] + tokenized_answers[i] for i in range(batch_size)]
            full_input_ids = [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in full_input_ids]
            confidence_estimation_inputs = self.tokenizer.pad(full_input_ids, return_attention_mask=True, return_tensors="pt").to(self.model.device)
            
            model_outputs = self.model(**confidence_estimation_inputs)
            logits = model_outputs.calibrated_logits if hasattr(model_outputs, "calibrated_logits") else model_outputs.logits
            
            for i in range(batch_size):
                prediction_logits = logits[i][-len(tokenized_answers[i]) - 1:-1].cpu()
                    
                prediction_token_ids = tokenized_answers[i]
                all_confidences.append(self.reduce_logits(prediction_logits, prediction_token_ids))
        return all_predicted, all_confidences, prediction_logits.detach().tolist() # [i.item() for i in all_confidences.cpu()]

    @torch.no_grad()
    def batch_inference_frq_multiple(self, batch_input: Dict[str, torch.Tensor], stop_generation_regex="\n", **generation_kwargs) -> List[str]:
        generated_sequences = self.model.generate(**batch_input, **generation_kwargs, num_return_sequences=5)
        all_predicted = []
        for i in range(len(generated_sequences)):
            sample_predictions = []
            for j in range(len(generated_sequences[i])):
                tokenized_question_len = len(batch_input["input_ids"][i])
                tokenized_prediction = generated_sequences[i][j]
                tokenized_prediction = tokenized_prediction[tokenized_question_len:].cpu().tolist()
                string_prediction = self.tokenizer.decode(tokenized_prediction, skip_special_tokens=True)

                string_prediction = string_prediction.lstrip()
                stop_indices = [int(m.start(0)) for m in re.finditer(stop_generation_regex, string_prediction)]
                stop_idx = stop_indices[0] if len(stop_indices) > 0 else len(string_prediction)
                string_prediction = string_prediction[:stop_idx]
                sample_predictions.append(string_prediction)
            all_predicted.append(sample_predictions)
        return all_predicted

    def get_softmax_probs(self, all_token_probs, choices):
        if self.eval_args.prechoice_softmax:
            probs = torch.nn.functional.softmax(all_token_probs / self.eval_args.temperature_scalar, dim=-1)
            choice_probs = probs[:, choices].detach()
        else:
            choice_probs = torch.nn.functional.softmax(all_token_probs[:, choices] / self.eval_args.temperature_scalar, dim=-1).detach()
        return choice_probs

    def get_subjects(self):
        if "token" in inspect.signature(cached_file).parameters:
            kwargs = {"token": self.model_args.hf_hub_token}
        elif "use_auth_token" in inspect.signature(cached_file).parameters: # for transformers==4.31.0
            kwargs = {"use_auth_token": self.model_args.hf_hub_token}
        
        if os.path.isdir(os.path.join(self.eval_args.task_dir, self.eval_args.task)):
            mapping_file = cached_file(
                path_or_repo_id=os.path.join(self.eval_args.task_dir, self.eval_args.task),
                filename="mapping.json",
                cache_dir=self.model_args.cache_dir,
                **kwargs
            )
            with open(mapping_file, "r", encoding="utf-8") as f:
                subjects: Dict[str, Dict[str, str]] = json.load(f)
            return subjects
        else:
            mapping_file = cached_file(
                path_or_repo_id=self.eval_args.task_dir, 
                filename="single_dataset_mapping.json",
                cache_dir=self.model_args.cache_dir,
                **kwargs
            )
            with open(mapping_file, "r", encoding="utf-8") as f:
                all_subjects: Dict[str, Dict[str, str]] = json.load(f)
            return {k:v for k,v in all_subjects.items() if k == self.eval_args.task}

    def get_eval_dataset(self, subject):
        if os.path.exists(os.path.join(self.eval_args.task_dir, self.eval_args.task)):
            return load_dataset(
                path=os.path.join(self.eval_args.task_dir, self.eval_args.task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token
            )
        else:
            return load_dataset(
                path=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token
            )
        return subjects
    
    def eval(self) -> None:
        subject_metrics = {}
        subjects = self.get_subjects()
        pbar = tqdm(subjects.keys(), desc="Processing subjects", position=0)
        all_has_categories = True
        
        subject_idx = 0
        for subject in pbar:

            subject_idx += 1
            dataset = self.get_eval_dataset(
                subjects[subject]["dataset_path"] if subjects[subject].get("dataset_path") is not None else subject
            )
            pbar.set_postfix_str(subjects[subject]["name"])
            inputs, predictions, labels, confidences, logits, entropies = [], [], [], [], [], []
            for i in trange(0, len(dataset[self.data_args.split]), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False):

                batch_inputs, batch_queries, batch_support_set = [], [], []
                for idx in range(i, min(i + self.eval_args.batch_size, len(dataset[self.data_args.split]))):
                    if self.eval_args.n_shot > 0:
                        support_set = dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                    else:
                        support_set = []
                    query, resp, history = self.eval_template.format_example(
                        target_data=dataset[self.data_args.split][idx],
                        support_set=support_set,
                        subject_name=subjects[subject]["name"],
                        use_history=self.template.use_history
                    )
                    if type(resp) is bool:
                        resp_to_encode = str(resp)
                    else:
                        resp_to_encode = resp if type(resp) is str else resp[0]

                    input_ids, _ = self.template.encode_oneturn(
                        tokenizer=self.tokenizer, query=query, resp=resp_to_encode, history=history
                    )
                    inputs = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
                    labels.append(resp)
                    batch_inputs.append(inputs)
                    batch_queries.append(dataset[self.data_args.split][idx]["question"])
                    batch_support_set.append(support_set)

                batch_inputs = self.tokenizer.pad(
                    batch_inputs, return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                
                if self.eval_args.task_type == "mcq":
                    batch_predictions, batch_confidences, batch_logits = self.batch_inference_mcq(batch_inputs)
                else:
                    batch_predictions, batch_confidences, batch_logits = self.batch_inference_frq(batch_inputs, self.eval_args.stop_generation_regex, **self.generation_kwargs)

                predictions += batch_predictions
                confidences += batch_confidences

            metrics_dict = {}
            subject_category = subjects[subject].get("category")
            if subject_category is not None:
                metrics_dict["category"] = subjects[subject]["category"]
            else:
                all_has_categories = False
            metrics_dict["subject"] = subject
            metrics_dict["predictions"] = predictions
            metrics_dict["labels"] = labels
            metrics_dict["confidences"] = confidences
            metrics_dict["logits"] = logits
            metrics_dict["entropies"] = entropies

            questions = list(dataset[self.data_args.split]["question"])
            metrics_dict["corrects"] = self.correctness_fn(predictions, labels[:len(predictions)], questions[:len(predictions)])

            metrics_dict = add_metrics(metrics_dict)
            subject_metrics[subject] = metrics_dict
            
            
        # aggregate results into categories
        category_metrics = None
        if all_has_categories:
            category_metrics = aggregate_metrics(subject_metrics, category_key="category")
            for k in category_metrics:
                category_metrics[k] = add_metrics(category_metrics[k])
        # aggregate all results
        all_metrics = aggregate_metrics(subject_metrics, category_key=None)
        all_metrics["Average"] = add_metrics(all_metrics["Average"])
        pbar.close()
        self._save_results(subject_metrics, all_metrics, category_metrics)
    

    def _save_results(self, subject_metrics, all_metrics, category_metrics=None) -> None:
        score_info = ""
        score_info += format_score_info(all_metrics)
        if category_metrics is not None:
            score_info += format_score_info(category_metrics)
        print(score_info)
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=True)
            with open(os.path.join(self.eval_args.save_dir, "subject_metrics.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(subject_metrics, f, indent=2)
            with open(os.path.join(self.eval_args.save_dir, "all_metrics.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(all_metrics, f, indent=2)
            if category_metrics is not None:
                with open(os.path.join(self.eval_args.save_dir, "category_metrics.json"), "w", encoding="utf-8", newline="\n") as f:
                    json.dump(category_metrics, f, indent=2)
            with open(os.path.join(self.eval_args.save_dir, "score_info.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)
            # Make ece plot
            create_ece_plot(
                all_metrics["Average"]["confidence_bin_values"],
                all_metrics["Average"]["accuracy_bin_values"],
                all_metrics["Average"]["bin_counts"],
                output_file=os.path.join(self.eval_args.save_dir, "average_ece_plot.png")
            )
            if all_metrics["Average"].get("best_confidence_bin_values") is not None:
                create_ece_plot(
                    all_metrics["Average"]["best_confidence_bin_values"],
                    all_metrics["Average"]["best_accuracy_bin_values"],
                    all_metrics["Average"]["best_bin_counts"],
                    output_file=os.path.join(self.eval_args.save_dir, "average_best_ece_plot.png")
                )
            if category_metrics is not None:
                for category_name, cm in category_metrics.items():
                    create_ece_plot(
                        cm["confidence_bin_values"],
                        cm["accuracy_bin_values"],
                        cm["bin_counts"],
                        output_file=os.path.join(self.eval_args.save_dir, f"{category_name.lower()}_ece_plot.png")
                    )

                    if cm.get("best_confidence_bin_values") is not None:
                        create_ece_plot(
                            cm["best_confidence_bin_values"],
                            cm["best_accuracy_bin_values"],
                            cm["best_bin_counts"],
                            output_file=os.path.join(self.eval_args.save_dir, f"{category_name.lower()}_best_ece_plot.png")
                        )

if __name__ == "__main__":
    evaluator = CalibrationEvaluator()
    evaluator.eval()
