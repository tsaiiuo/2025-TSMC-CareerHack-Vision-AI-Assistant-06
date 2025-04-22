import requests
import time
import utils
import json
import yaml
import numpy as np
from PIL import Image

import vertexai
from vertexai.generative_models import GenerativeModel

import google.generativeai as genai


PROJECT_ID = "tsmccareerhack2025-aaid-grp6"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

# utility function
def format_messages(messages: list):
    # current attempt is to wrap the messages in a dict structure
    messages_content_only = [message['role']+':'+message["content"] for message in messages]
    # return json.dumps(messages_content_only)
    m_all = ''
    for m in messages_content_only:
        m_all+=f'{m}\n'
    return m_all  

class LLM():
    def __init__(self, model_name="gemini-1.5-pro"):
        self.model_name = model_name

    def generate(self, messages, temperature=0):
        # wrapping the messages in a dict; doing this separately between system prompt and user prompt
        system_messages = [system_message for system_message in messages if system_message["role"] == "system"]
        user_messages = [user_message for user_message in messages if user_message["role"] == "user"]
        
        system_message_str = format_messages(system_messages)
        user_message_str = format_messages(user_messages)
        
        model = GenerativeModel(
          model_name="gemini-1.5-pro-002",
          system_instruction=system_message_str)
        
        # genai.configure(api_key="XXXX")
        # model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_message_str)
        
        # tic = time.time()
        # print(user_message_str)
        response = model.generate_content(user_message_str, generation_config={"temperature": temperature})
        
        # debug
        # print('\n\n\n')
        # print('LLM Response:')
        # print('-----------------------')
        # print(response.text)
        # print('-----------------------')
        # print('\n\n\n')
        
        returned_dict = dict()
        returned_dict["choices"] = [dict()]
        returned_dict["choices"][0]["message"] = dict()
        returned_dict["choices"][0]["message"]["content"] = response.text
        
        # toc = time.time()
        # print(f'Time: {toc-tic}s')
        
        return returned_dict

