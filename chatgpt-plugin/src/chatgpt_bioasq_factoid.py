import os
import sys
import openai
import jsonlines
import json
import copy
from tqdm import tqdm
import time
import random
random.seed(1234)

from chatgpt_api import get_response
from datasets import load_dataset
import numpy as np


base_prompt = [{
"role": "system",
"content": "You are a helpful assistant that can answer biomedical question after the \"**Question**:\" based on the provided context after the \"**Context**:\". Please only return the words in \"**Context**:\" after \"**Answer**:\"."
},
{
"role": "user",
"content": ""
},
{
"role": "assistant",
"content": "**Answer**: "
}]


plugin_prompt = [{
"role": "system",
"content": "You are a helpful assistant that can answer biomedical question after the \"**Question**:\" based on the provided context after the \"**Context**:\". Please only return the words in \"**Context**:\" after \"**Answer**:\". The results after \"**BioLinkBert Prediction**\" are for reference, where confidence represents the probability predicted by the model."
},
{
"role": "user",
"content": ""
},
{
"role": "assistant",
"content": "**Answer**: "
}]


def read_dataset(one_dataset, is_train=False, max_len=1000):
    all_examples = {}
    for idx in range(len(one_dataset)):
        context = one_dataset[idx]["context"]
        question = one_dataset[idx]["question"]
        answer = one_dataset[idx]["answers"]["text"][0]
        example_id = one_dataset[idx]["id"]
        if is_train and len(context.split(" ")) > max_len:
            continue
        all_examples[example_id] = {"context": context, "question": question, "ans": answer}
    return all_examples


def read_cls(data_files):
    raw_datasets = load_dataset("json", data_files=data_files)

    train_examples = read_dataset(raw_datasets["train"], is_train=True)
    test_examples = read_dataset(raw_datasets["test"])
    
    return train_examples, test_examples


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def read_cls_plugin(file_name):
    all_examples = {}
    with open(file_name, "r") as f:
        all_data = json.load(f)
        for example_id in all_data:
            all_examples[example_id] = f"{all_data[example_id][0]['text']} (Confidence: {all_data[example_id][0]['probability']:.2f})"
    return all_examples


def run_cls(file_name, all_examples, train_examples, k=5):
    all_res = []
    with jsonlines.open(file_name, "w") as f:
        for idx in tqdm(all_examples):
            context_examples = random.sample(list(train_examples.keys()), k)
            prompt = [copy.deepcopy(base_prompt[0])]
            for x in context_examples:
                prompt.extend(copy.deepcopy(base_prompt[1:]))
                prompt[-2]["content"] = "**Context**: " + train_examples[x]["context"] + "\n"
                prompt[-2]["content"] += "**Question**: " + train_examples[x]["question"]
                prompt[-1]["content"] += train_examples[x]["ans"] + "."
            prompt.extend(copy.deepcopy(base_prompt[1:]))
            prompt[-2]["content"] = "**Context**: " + all_examples[idx]["context"] + "\n"
            prompt[-2]["content"] += "**Question**: " + all_examples[idx]["question"]

            chatgpt_response = get_response(prompt)
            if chatgpt_response == "error":
                print("get num:", len(all_res))
                break
            res_example = {"answer": all_examples[idx]["ans"]}
            res_example["chatgpt"] = chatgpt_response["choices"][0]["message"]["content"]
            res_example["prompt"] = prompt
            f.write(res_example)
            all_res.append(chatgpt_response)
            # break


def run_cls_plugin(file_name, all_examples, train_examples, plugin_test_examples, plugin_train_examples, k=5):
    all_res = []
    with jsonlines.open(file_name, "w") as f:
        for idx in tqdm(all_examples):
            context_examples = random.sample(list(train_examples.keys()), k)
            prompt = [copy.deepcopy(base_prompt[0])]
            for x in context_examples:
                prompt.extend(copy.deepcopy(base_prompt[1:]))
                prompt[-2]["content"] = "**Context**: " + train_examples[x]["context"] + "\n"
                prompt[-2]["content"] += "**Question**: " + train_examples[x]["question"] + "\n"
                prompt[-2]["content"] += "**BioLinkBert Prediction**: " + plugin_train_examples[x] + "."
                prompt[-1]["content"] += train_examples[x]["ans"] + "."
            prompt.extend(copy.deepcopy(base_prompt[1:]))
            prompt[-2]["content"] = "**Context**: " + all_examples[idx]["context"] + "\n"
            prompt[-2]["content"] += "**Question**: " + all_examples[idx]["question"] + "\n"
            prompt[-2]["content"] += "**BioLinkBert Prediction**: " + plugin_test_examples[idx] + "."

            chatgpt_response = get_response(prompt)
            if chatgpt_response == "error":
                print("get num:", len(all_res))
                break
            res_example = {"answer": all_examples[idx]["ans"], "ref_answer": plugin_test_examples[idx].split(" (Confidence")[0]}
            res_example["chatgpt"] = chatgpt_response["choices"][0]["message"]["content"]
            res_example["prompt"] = prompt
            f.write(res_example)
            all_res.append(chatgpt_response)
            # break


def eval_cls(file_name):
    correct_num = 0
    all_num = 0
    error_num = 0
    exact_num = 0
    ref_num = 0
    with open(file_name, "r") as f:
        for line in tqdm(f.readlines()):
            elements = json.loads(line.strip())
            all_num += 1
            if elements["answer"].lower().replace(".", "") == elements["chatgpt"].lower().replace(".", ""):
                exact_num += 1
            if elements["answer"].lower().replace(".", "") in elements["chatgpt"].lower().replace(".", ""):
                correct_num += 1
            if "ref_answer" in elements:
                if elements["answer"].lower().replace(".", "") == elements["ref_answer"].lower().replace(".", ""):
                    ref_num += 1
    print("all_error:", error_num)
    print("exact res:", exact_num, all_num, exact_num / all_num)
    print("in res:", correct_num, all_num, correct_num / all_num)
    print("ref res:", ref_num, all_num, ref_num / all_num)


if __name__ == "__main__":
    data_path = sys.argv[1]
    prediction_path = sys.argv[2]
    icl_num = sys.argv[3]
    output_path = sys.argv[4]

    data_files = {"train": data_path + "/train.json", "test": data_path + "/test.json"}
    prediction_files = {"train": data_path + "/train_nbest_predictions.json", "test": data_path + "/predict_nbest_predictions.json"}

    train_examples, test_examples = read_cls(data_files)
    plugin_train_examples = read_cls_plugin(prediction_files["train"])
    plugin_test_examples = read_cls_plugin(prediction_files["test"])

    run_cls_plugin(output_path, test_examples, train_examples, plugin_test_examples, plugin_train_examples, k=icl_num)
    eval_cls(output_path)