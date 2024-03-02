import os
import openai
import time
import random


openai.api_key = "" # The openai api key to use chatgpt


def get_response(prompt, all_error=0):
    model_name = "gpt-3.5-turbo-0613"
    # model_name = "gpt-3.5-turbo-16k-0613"

    try:
        chatgpt_response = openai.ChatCompletion.create(
            model=model_name,
            messages=prompt,
            temperature=0.2
        )
        # time.sleep(1)
    except Exception as e:
        print("Inference error, sleep 30 seconds")
        time.sleep(30)
        # print(e)
        if all_error < 5: # rerun 5 times
            chatgpt_response = get_response(prompt, all_error + 1)
        else:
            return "error"
    return chatgpt_response


if __name__ == "__main__":
    prompt = [{
    "role": "system",
    "content": ""
    },
    {
    "role": "user",
    "content": "Hello"
    },
    {
    "role": "assistant",
    "content": ""
    }]

    try:
        chatgpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0.2
        )
        # time.sleep(1)
        print(chatgpt_response)
    except Exception as e:
        print(e)
