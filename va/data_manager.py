import base64
from io import BytesIO
from PIL import Image
from tools import FUNCTIONS
import json

from colorama import Fore, Style
from prompt.planner_prompt import (
    PLAN,
    EXAMPLE_PLAN1,
    EXAMPLE_PLAN2,
    EXAMPLE_PLAN3,
    EXAMPLE_PLAN4,
    EXAMPLE_PLAN5,
    EXAMPLE_PLAN6,
    EXAMPLE_PLAN7,
    KNOWLEGE
    )

import utils

class DataManager:
    def __init__(self, data_frame = None, data_root = None, debug = False):
        self.table = data_frame
        self.data_root = data_root
        self.debug = debug

    def __getitem__(self, i):
        ob = self.table.iloc[i,:]
        return self.get_messages(ob)

    def __len__(self):
        return self.table.shape[0]

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def get_messages(self, ob):
        user_prompt, artifact = self.get_user_prompt(ob)
        sys_prompt = self.get_sys_prompt()
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self.print(Fore.WHITE + user_prompt)
        self.print(Style.RESET_ALL)
        return messages, artifact, ob

    def get_sys_prompt(self):
        tool_doc = self.get_tool_doc()
        sys_prompt = PLAN.format(
            tool_desc = tool_doc,
            examples = f"{EXAMPLE_PLAN1}\n{EXAMPLE_PLAN2}\n{EXAMPLE_PLAN3}\n{EXAMPLE_PLAN4}\n{EXAMPLE_PLAN5}\n{EXAMPLE_PLAN6}\n{EXAMPLE_PLAN7}\n",
            knowlege = KNOWLEGE
            )
        return sys_prompt

    def get_tool_doc(self):
        available_functions = {n:f for n, f in FUNCTIONS.__dict__.items() if not n.startswith("_") and callable(f)}
        tool_doc = ""
        for f in available_functions:
            doc = available_functions[f].__doc__
            tool_doc += f"{f}: {doc}\n"
        return tool_doc

    def get_artifact_prompt(self, ob):
        artifact = []
        artifact_type = []
        artifact_name = []
        artifact_path = []
        inputs = eval(ob['input'])
        for k in inputs:
            path = self.data_root+'/'+inputs[k]
            artifact.append(Image.open(path))
            artifact_path.append(path)
            artifact_name.append(k)
            self.print(path)
        artifact_type = [type(i) for i in (artifact)]
        artifact_prompt = f"Input is a list of object which type is {artifact_type}, the length of the list is {len(artifact)}, the name of objects is {artifact_name}."
        return artifact_prompt, artifact_path

    def get_user_prompt(self, ob):
        artifact_prompt, artifact_path = self.get_artifact_prompt(ob)
        user_prompt = ob['detail']
        
        user_prompt = f"<step 0> USER: {user_prompt}\n{artifact_prompt}"
        return user_prompt, artifact_path

    

class DataManagerService(DataManager):
    def get_artifact_prompt(self, ob):
        # inputs = utils.convert_base64_to_image(ob["input"]["base64"])
        if isinstance(ob["input"], str):
            artifacts = json.loads(ob["input"])
        else:
            artifacts = ob["input"]
        # Extract artifact types and names using list comprehensions
        artifact_type = [item["type"] for item in artifacts]
        artifact_name = [item["name"] for item in artifacts]
        
        # Convert base64 strings to images
        artifact = [utils.convert_base64_to_image(item["base64"]) for item in artifacts]
        # artifact_type = [type(i) for i in (artifact)]
        
        # Create the artifact prompt
        artifact_prompt = f"Input is a list of object which type is {artifact_type}, the length of the list is {len(artifact)}, the name of objects is {artifact_name}."
        return artifact_prompt, artifact





