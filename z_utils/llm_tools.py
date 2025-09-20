from openai import OpenAI
import os
import base64
from dotenv import load_dotenv
from termcolor import colored

load_dotenv()


def image_to_base64_data_url(image_path):
    with open(image_path, "rb") as image_file:
        base64_encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_encoded_image}"


base_url = os.getenv("BASE_URL")
model_name = os.getenv("MODEL")
api_key = os.getenv("API_KEY")

base_url = os.getenv("VLM_BASE_URL")
model_name = os.getenv("VLM_MODEL")
api_key = os.getenv("VLM_API_KEY")


client = OpenAI(api_key=api_key, base_url=base_url)


pic_path = "no_git_oic/image.png"

messages = [{"role": "user", "content": "Hello, how are you? Who are you?"}]
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_to_base64_data_url(pic_path)},
            },
            {"type": "text", "text": "对图片进行OCR,输出markdown格式"},
        ],
    }
]


response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.7,
)

# uv run z_utils/llm_tools.py
print(colored(f"{response.choices[0].message.content}", "light_yellow"))

""" 
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "1+1=?"},
]

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"
                },
            },
            {"type": "text", "text": "What is the text in the illustrate?"},
        ],
    },
]
"""
