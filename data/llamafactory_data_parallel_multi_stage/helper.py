import re

from data_postprocessing.utils import read_json
from tqdm import tqdm


RAW_FILE = "/data1/gsn/CustomTrain/data/88repo_1000_no_lsp/llamafactory_data_parallel_multi_stage/stage_scale_2/train.json"

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

if __name__ == '__main__':
    check_input_and_output_lines()
