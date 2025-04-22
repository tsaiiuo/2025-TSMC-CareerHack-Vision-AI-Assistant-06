from colorama import Fore, Style
import time
from api import LLM
import traceback
from PIL import Image
import pandas as pd
import os
from typing import TypedDict
import numpy as np
import multiprocessing

from tools import FUNCTIONS
from utils_parallel import MultiController
import utils

class Result(TypedDict):
    id: str
    result: str

class Record():
    def __init__(self):
        self.obj_list = multiprocessing.Manager().list()
        self.timeout_name = 'timeout'

    def write(self, obj_id, result):
        result: Result = {
        "id": obj_id,
        "result": result
        }
        index = np.where([i['id']==obj_id for i in self.obj_list])[0]
        index = int(index)
        final = self.obj_list[index]
        if  final['id'] == obj_id and  final['result']==self.timeout_name:
            self.obj_list.pop(index)
        self.obj_list.append(result)

    def write_timeout(self, obj_id):
        result: Result = {
        "id": obj_id,
        "result": self.timeout_name
        }
        self.obj_list.append(result)
    
    def dump_record(self, output_path):
        record  = pd.DataFrame(list(self.obj_list))
        record.to_csv(output_path)

class VisionAssistant(MultiController):
    def __init__(self, step_limit=10, debug=False, timeout = 1000000, output_root = None, is_thread = False, memory_limit_mb = 100):
        super().__init__(is_thread = is_thread, timeout = timeout, memory_limit_mb = memory_limit_mb)
        self.llm = LLM()
        self.step_limit = step_limit
        self.debug = debug
        self.output_root = output_root
        self.record = Record()
        self.thinking_log = []
    def log_thinking(self, message):
        """Append a thinking message to the log and optionally print it."""
        self.thinking_log.append(message)
        if self.debug:
            print(message)

    def get_thinking_log(self):
        """Return all stored thinking messages as a single string (or as a list)."""
        return "\n".join(self.thinking_log)
        # Or simply: return self.thinking_log

    def extract_tag(self, content, tag):
        inner_content = None
        remaning = content
        all_inner_content = []

        while f"<{tag}>" in remaning:
            inner_content_i = remaning[remaning.find(f"<{tag}>") + len(f"<{tag}>") :]
            if f"</{tag}>" not in inner_content_i:
                break
            inner_content_i = inner_content_i[: inner_content_i.find(f"</{tag}>")]
            remaning = remaning[remaning.find(f"</{tag}>") + len(f"</{tag}>") :]
            all_inner_content.append(inner_content_i)

        if len(all_inner_content) > 0:
            inner_content = "\n".join(all_inner_content)
        return inner_content

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def write_record_result(self, shared_dict, result, info={}):
        obj_id = info.get('id', None)
        ret = None
        try:
            if isinstance(result, Image.Image):
                if self.output_root != None:
                    output_path = f'{self.output_root}/{obj_id}'
                    utils.check_dir(output_path)
                    output_path = f'{output_path}/result.png'
                    result.convert('L').save(output_path)
                    ret = output_path
                else:
                    ret = 'No image.'
            elif result is None:
                ret = 'No image.'
            else:
                ret = float(result)
        except:
            print(f'Fail on {obj_id}')
            ret = 'Fail'
        shared_dict['record'].write(obj_id, ret)

    def write_record_init(self, shared_dict, info={}):
        obj_id = info.get('id', None)
        shared_dict['record'].write_timeout(obj_id)

    def dump_record(self, output_root=None):
        if output_root is None:
            output_path = os.path.join(self.output_root, 'result.csv')
        else:
            output_path = os.path.join(output_root, 'result.csv')
        self.record.dump_record(output_path)
    
    def put_shared_dict(self, shared_dict):
        shared_dict['record'] = self.record
        return shared_dict

    def predict(self, messages, artifact_path, info={}):
        shared_dict = {'record': self.record}
        self.write_record_init(shared_dict, info)
        result = self._predict(messages, artifact_path)
        self.write_record_result(shared_dict, result, info)
        return result


    def predict_local(self, shared_dict, lock, messages, artifact_path, info={}):
        with lock:
            self.write_record_init(shared_dict, info)
            # print(shared_dict['record'].obj_list)
        result = self._predict(messages, artifact_path)
        with lock:
            self.write_record_result(shared_dict, result, info)

    def get_artifact(self, artifact_path):
        artifact = []
        for path in artifact_path:
            artifact.append(Image.open(path))
        return artifact

    def register_tool(self, namespace):
        available_functions = {n:f for n, f in FUNCTIONS.__dict__.items() if not n.startswith("_") and callable(f)}
        for f in available_functions:
            namespace[f] = available_functions[f]
        return namespace

    def _predict(self, messages, artifact_path):
        namespace = dict(globals())
        namespace = self.register_tool(namespace)
        if isinstance(artifact_path[0], str):
            namespace['INPUT'] = self.get_artifact(artifact_path)
        else:
            namespace['INPUT'] = artifact_path
        step = 0
        self.print(Fore.YELLOW + messages[1]['content'])
        self.log_thinking(messages[1]['content'])
        while True:
            ''' Inferecne '''
            response = self.llm.generate(messages=messages)
            step +=1
            if step >= self.step_limit: break
            response_msg = response["choices"][0]["message"]["content"]
            # with open(f"output/response_{step}.txt", 'w') as f:
            #     f.write(response_msg)

            thinking = self.extract_tag(response_msg, 'thinking')
            execute_python = self.extract_tag(response_msg, 'execute_python')
            finalize_plan = self.extract_tag(response_msg, 'finalize_plan')
            print(len(messages))
            messages.append({"role": "user", "content": f"<step {step}> AGENT: {response_msg}"})


            if finalize_plan:
                self.print(Fore.YELLOW + finalize_plan)
                self.log_thinking(finalize_plan)
                self.print(Style.RESET_ALL)
                try:
                    if isinstance(namespace['result'], Image.Image) and self.debug:
                        # namespace['result'].show()
                        namespace['result'].save('./IMAGE.png')
                    return namespace['result']
                except Exception as e:
                    pass
                break

            # self.print the thinking process
            self.print(Fore.GREEN + response_msg)
            self.log_thinking(response_msg)
            self.print(Style.RESET_ALL)

            if execute_python:
                # execute_python = execution_check(execute_python)
                try:
                    self.print(Fore.CYAN + f" -- running {execute_python}")
                    self.log_thinking(f" -- running {execute_python}")
                    self.print(Style.RESET_ALL)
                    # Apply available tools for the function execution
                    if globals() is not locals():
                        namespace.update(locals())
                    exec(execute_python, namespace)
                    obervation =  namespace['result']
                    if len(f"Observation: {obervation}") < 150:
                        self.print(Fore.BLUE + f"Observation: {obervation}")
                        self.log_thinking(f"Observation: {obervation}")
                    else:
                        self.print(Fore.BLUE + f"Observation: Too long")
                        self.log_thinking(f"Observation: Too long")
                        obervation = 'A veriable save in result.'
                        # breakpoint()
                    self.print(Style.RESET_ALL)
                    messages.append({"role": "user", "content": f"<step {step}> OBSERVATION: {obervation}"})
                    
                except Exception as e:
                    # raise NotImplementedError
                    error_class = e.__class__.__name__
                    detail = e.args[0]
                    tb = e.__traceback__
                    last_call_stack = traceback.extract_tb(tb)[-1]
                    # line = execute_python.splitlines()[last_call_stack.lineno-1]
                    error_message = "OBSERVATION: Error on line \"{} \": {}: {}".format(last_call_stack.lineno, error_class, detail)
                    self.print(Fore.RED + error_message)
                    self.log_thinking(error_message)
                    self.print(Style.RESET_ALL)
                    messages.append({"role": "user", "content": error_message})
            # If no action is detected then try to ask user to provide some feedback
            else:
                self.print(Fore.RED + f"No action is detected, you can give some feedback by input")         
                self.log_thinking(f"No action is detected, you can give some feedback by input")
                
                break
        return None