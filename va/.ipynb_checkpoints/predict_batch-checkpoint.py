import time
from PIL import Image
import pandas as pd

from utils import check_dir
from data_manager import DataManager
from agent.vision_assistant_agent import VisionAssistant

import threading
import time

if __name__ == '__main__':
    csv_path = f'../data/release_private_set.csv'
    # csv_path = f'../data/release_private_set.csv'
    data_root = '..'
    output_root =  '../output'
    check_dir(output_root)
    db = pd.read_csv(csv_path)
    db = DataManager(db, data_root)
    va = VisionAssistant(debug=False, timeout=120, output_root=output_root, memory_limit_mb = 500)

    count = 0
    for messages, artifact, ob in db:
        # if count > 2: break
        # result = va.predict(messages, artifact)
        va.add_task(messages, artifact, ob)
        count+=1



    tic = time.time()
    va.start_task(12)
    toc=  time.time()
    print(f'Done {count} tasks in {round(toc-tic, 3)} sec.')
    va.dump_record()

