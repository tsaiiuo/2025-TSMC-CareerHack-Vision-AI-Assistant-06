import threading
import multiprocessing
import utils
from tqdm import tqdm, trange
import time
import os
import psutil


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        return thread
    return wrapper

def process(fn):
    def wrapper(*args, **kwargs):
        process = multiprocessing.Process(target=fn, args=args, kwargs=kwargs)
        return process
    return wrapper


class MultiController(object):
    def __init__(self, is_thread = False, timeout=1000000, memory_limit_mb = 100):
        self.memory_limit_mb = memory_limit_mb
        self.is_thread = is_thread
        if is_thread:
            self.task_creator = threading.Thread
        else:
            self.task_creator = multiprocessing.Process
        self.jobs = {}
        self.task_count = 0
        self.timeout = timeout

    def put_shared_dict(self, shared_dict):
        shared_dict['test'] = 1
        return shared_dict

    def predict_local(self, shared_dict, lock, *args, **kwargs):
        with lock:
            print(shared_dict['test'])
            shared_dict['test']+=1
        return None

    def add_task(self, *args, **kwargs):
        self.task_count += 1
        self.jobs[self.task_count] = {
            'target': self.predict_local,
            'args': args
        }

    def start_task(self, multi_count = 16):
        self.start(multi_count = multi_count) 

    def start_queue(self, jobs_queue):
        for job in jobs_queue:
            job.start()
        if not self.is_thread:
            self.monitor_process(jobs_queue, memory_limit_mb=self.memory_limit_mb)
        for job in jobs_queue:
            job.join(self.timeout)
    
    def start_all(self):
        # task = self.task_creator(target=self.predict_local, args=args, kwargs=kwargs)
        jobs = []
        for key, values in self.jobs.items():
            target = values.get('target', None)
            args = values.get('args', ())
            kwargs = values.get('kwargs', {})
            task = self.task_creator(target=target, args=args, kwargs=kwargs)
            jobs.append(task)
        if self.is_thread:
            task.setDaemon(True) # force terminate when the main thread stops
        else:
            task.daemon = True  

        for job in jobs:
            job.start()

    def start(self, multi_count):
        count = 0
        jobs_queue = []

        if self.is_thread:
            lock = threading.Lock()
            shared_dict = {}
        else:
            lock = multiprocessing.Lock()
            shared_dict = multiprocessing.Manager().dict()
        shared_dict = self.put_shared_dict(shared_dict)
        for job_id, values in tqdm(self.jobs.items()):
            target = values.get('target', None)
            args = values.get('args', ())
            kwargs = values.get('kwargs', {})
            args = (lock,) + args
            args = (shared_dict,) + args
            task = self.task_creator(target=target, args=args, kwargs=kwargs)
            jobs_queue.append(task)
            count+=1
            if count % multi_count == 0:
                self.start_queue(jobs_queue)
                jobs_queue = []
                count = 0
        if jobs_queue:
             self.start_queue(jobs_queue)
        self.jobs = {}
        self.task_count = 0

    def check_alive(self, proc_list):
        alive_list = [p.is_alive() for p in proc_list]
        return any(alive_list)

    def monitor_process(self, proc_list, memory_limit_mb):
        is_break = False
        while self.check_alive(proc_list):
            for proc in proc_list:
                try:
                    mem_usage = psutil.Process(proc.pid).memory_info().rss / 1024**2  # lb:MB
                except Exception as e:
                    mem_usage = 0
                # print(f"Memory usage: {mem_usage:.2f} MB")
                if mem_usage > memory_limit_mb:
                    print("Memory limit exceeded, terminating process.")
                    proc.terminate()
                    is_break = True
            if is_break:
                break
            time.sleep(1)


# if __name__ == '__main__':
#     a = MultiController()
#     for i in range(3):
#         a.add_task()
#     a.start_task()
#     # a.start_all()