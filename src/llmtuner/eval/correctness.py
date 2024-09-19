import numpy as np
import re
import string

import os
import openai
from tqdm import tqdm

openai.api_type = "azure"
openai.api_base = "https://gpt-test-2-aiservices447677599.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

def openai_language_model_evaluation(prediction, label, question, engine="gpt-35-turbo"):
    # Just select the first of all candidate answers for the "gold" answer
    str_label = "\n".join(label)
    if label is not None:
        prompt = ("""Are the following two answers to my question Q semantically equivalent?"""
        + """\n\nQ:{question}\nA1: {label}\nA2: {prediction}\n\n"""
        + """Please answer with a single word, either “Yes.“ or “No.“""")
        
        message_text = [
            {"role":"system","content": prompt.format(**{"question": question, "label": label, "prediction": prediction})}
        ]
    else:
        prompt = ("""Is the following answer A to my question Q both truthful and helpful?"""
        + """\n\nQ:{question}\nA: {prediction}\n\n"""
        + """Please answer with a single word, either “Yes“ or “No“""")
        message_text = [
            {"role":"system","content": prompt.format(**{"question": question, "label": label, "prediction": prediction})}
        ]
    #print(message_text)
    try:
        completion = openai.ChatCompletion.create(
            engine=engine,
            messages = message_text,
            temperature=0.0,
            max_tokens=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
    except Exception as e:
        print(e)
        return False
    chat_model_answer = completion["choices"][0].message["content"]
    chat_model_answer = chat_model_answer.lower()
    if chat_model_answer in ["yes", "no"]:
        return chat_model_answer == "yes"
    else:
        print("Chat model evaluation error, chat model did not output either yes or no.")
        return False

def openai_many_answer_evaluation(prediction, label, question, engine="gpt-35-turbo"):
    return any([openai_language_model_evaluation(prediction, l, question) for l in label])

def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    Added removing "Answer: "
    """
    
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_answer_prefix(text):
        return text.replace("answer ", "")

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_answer_prefix(remove_punc(lower(s)))))

def exact_match_fn(
    predictions,
    labels,
):
    correctness = []
    for prediction, possible_labels in zip(predictions, labels):
        normalized_prediction = normalize_answer(prediction)
        normalized_labels = [normalize_answer(l) for l in possible_labels]
        correctness.append(normalized_prediction in normalized_labels)

    return correctness

def get_correctness_fn(name, **kwargs):
    if name == "mcq":
        return lambda predictions, labels, questions: (np.array(predictions) == np.array(labels)).tolist()
    elif name == 'answerable':
        # Should return true if the label is answerable else false
        return labels #lambda predictions, labels, questions: [l == True for l in labels]
    elif name == "frq_em":
        return lambda predictions, labels, questions: exact_match_fn(predictions, labels, **kwargs)
    elif name == "frq_lm_gpt35":
        return lambda predictions, labels, questions: [openai_many_answer_evaluation(p, l, q, engine="gpt-35-turbo") for p, l, q in tqdm(zip(predictions, labels, questions), total=len(predictions))]
    elif name == "frq_lm_gpt4":
        return lambda predictions, labels, questions: [openai_many_answer_evaluation(p, l, q, engine="gpt-4") for p, l, q in tqdm(zip(predictions, labels, questions), total=len(predictions))]
    elif name == "frq_lm_no_label_gpt35":
        return lambda predictions, labels, questions: [openai_language_model_evaluation(p, None, q, engine="gpt-35-turbo") for p, q in tqdm(zip(predictions, questions), total=len(predictions))]
    elif name == "frq_lm_no_label_gpt4":
        return lambda predictions, labels, questions: [openai_language_model_evaluation(p, None, q, engine="gpt-4") for p, q in tqdm(zip(predictions, questions), total=len(predictions))]
    else:
        raise ValueError(f"Unknown correctness function {name}")