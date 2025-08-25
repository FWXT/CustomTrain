import random
import re

from data_postprocessing.utils import read_json, write_json
from tqdm import tqdm


RAW_FILE = "/data1/gsn/CustomTrain/data/coeditor-python/raw/stage_scale_4/train.json"
NEW_FILE = "/data1/gsn/CustomTrain/data/coeditor-python/example/train_5.json"

def check_input_and_output_lines():
    data = read_json(RAW_FILE)

    extra_id_pattern = r'<extra_id_\d+>'
    mismatch = 0
    for obj in tqdm(data, total=len(data)):
        input_text = obj['input']
        output_text = obj['output']

        input_num = len(re.findall(extra_id_pattern, input_text))
        output_num = len(re.findall(extra_id_pattern, output_text))
        if input_num != output_num:
            mismatch += 1

    print(f"{RAW_FILE} mismatch: {mismatch}/{len(data)}")

def shuffle_dataset():
    data = list(read_json(RAW_FILE))
    random.shuffle(data)
    write_json(data, NEW_FILE)

def extract_example_data():
    data = read_json(RAW_FILE)
    write_json(data[:5], NEW_FILE)

if __name__ == '__main__':
    # check_input_and_output_lines()
    # shuffle_dataset()
    extract_example_data()
