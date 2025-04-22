import pandas as pd
from data_manager import DataManager
from agent.vision_assistant_agent import VisionAssistant
import time
from utils import check_dir
import shutil

if __name__ == '__main__':
    csv_path = f'../data/release_public_set.csv'
    data_root = '..'
    output_root = '../output'

    db = pd.read_csv(csv_path)
    db = DataManager(db, data_root)
    # db = DataManager(db, data_root)
    va = VisionAssistant(debug=True, timeout=50, output_root=output_root, is_thread = True, memory_limit_mb = 150)
    tic = time.time()
    input_paths = None
    obj_id = None
    for messages, artifact, ob in db:
        
        if ob['id'] == "bd27cf15-c7fe-44b7-a7e7-9578f1a21088":
            result = va.predict(messages, artifact)
            # input_paths = artifact
            # print(messages[0])
            # obj_id = ob['id']
            # va.add_task(messages, artifact, ob)
            
    # va.start_task(1)
    # print(result)
    toc=  time.time()
    print(f'Done in {round(toc-tic, 3)} sec.')
    va.dump_record()
    output_path = f"../output/{obj_id}"
    for input_path in input_paths:
        name = input_path.split('/')[-1].split('.')[0]
        shutil.copy(input_path, f'{output_path}/{name}.png')