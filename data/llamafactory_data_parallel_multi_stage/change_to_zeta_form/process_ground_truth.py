import json
import os
from typing import List
from .common import _Tokenizer, tokens_to_change, inline_output_tokens, normalize_code_by_ast
from tqdm import tqdm

RAW_FILE = "/data/mnt_bucket/qzq/CustomTrain/data/llamafactory_data_parallel_multi_stage/change_to_zeta_form/random_5000_test_v2.json"
NEW_FILE = "/data/mnt_bucket/qzq/CustomTrain/data/llamafactory_data_parallel_multi_stage/change_to_zeta_form/random_5000_test_zeta.json"

DIRECTORY_PREFIX = "stage_scale"
RAW_FILE_PREFIX = "necessary_"

FILE_SUFFIX = "_zeta"

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, output_file: str):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def extract_main_input(input_text):
    """
    从input文本中提取main_input部分
    Args:
        input_text: 完整的input文本
    Returns:
        str: 提取的main_input部分
    """
    # start_marker = "========Main Code to Edit========"
    # end_marker = "========References========"
    start_marker = "### User Excerpt:"
    end_marker = "### User Edits Reference:"
    
    # 查找开始和结束标记的位置
    start_pos = input_text.find(start_marker)
    end_pos = input_text.find(end_marker)

    if end_pos == -1:
        end_marker = "### Response:"
        end_pos = input_text.find(end_marker)

    if start_pos == -1 or end_pos == -1:
        print(f"警告: 未找到指定的标记 {start_pos} {end_pos}")
        return None
    
    # 提取标记之间的内容
    main_section = input_text[start_pos + len(start_marker):end_pos]
    
    # 查找第一个<extra_id_开头的标签
    import re
    extra_id_pattern = r'<extra_id_\d+>'
    match = re.search(extra_id_pattern, main_section)
    
    if match:
        # 从第一个<extra_id_标签开始截取
        start_idx = match.start()
        main_input = main_section[start_idx:]
        return main_input.strip()
    else:
        print("警告: 未找到<extra_id_标签")
        return main_section.strip()

def get_code(input, ground_truth):
    """
    将输入文本转换为token_ids, 然后计算代码变化
    Args:
        input: 输入文本
        ground_truth: 真实答案
    Returns:
        tuple: label_code 标签代码
    """
    # 使用tokenizer将文本转换为token_ids
    input_tokens = _Tokenizer.encode(input, add_special_tokens=False)
    ground_truth_tokens = _Tokenizer.encode(ground_truth, add_special_tokens=False)
    
    label_code = tokens_to_change(
        inline_output_tokens(input_tokens, ground_truth_tokens)
    ).after
    return label_code

def is_unchanged_data(s: str) -> bool:
    return ("<add>" not in s) and ("<del>" not in s)

def process_ground_truth(data: List, output_file: str, add_unchanged_field: bool = True):
    total = len(data)
    for i, obj in tqdm(enumerate(data), total=total):
        full_input = obj["input"]
        ground_truth = obj["output"]

        if add_unchanged_field:
            obj["is_unchanged"] = is_unchanged_data(ground_truth)

        main_input = extract_main_input(full_input)
        if main_input is None:
            print(i)
            continue
        label_code = get_code(main_input, ground_truth).replace("</s>", "").replace("<s>", "")

        label_code = "<extra_id_start>\n" + label_code + "\n<extra_id_end>"
        obj["output"] = label_code
    
    write_json(data, output_file)

def main():
    data = read_json(RAW_FILE)
    process_ground_truth(data, NEW_FILE)

    # parent_dir = os.path.dirname(os.getcwd())
    # print(parent_dir, flush=True)
    # for path in os.listdir(parent_dir):
    #     abspath = parent_dir + "/" + path
    #     if os.path.isdir(abspath) and path.startswith(DIRECTORY_PREFIX):
    #         for filename in os.listdir(abspath):
    #             if filename.startswith(RAW_FILE_PREFIX):
    #                 raw_file = abspath + "/" + filename
    #                 basename = os.path.splitext(os.path.basename(filename))[0]
    #                 output_file = abspath + "/" + basename + FILE_SUFFIX + ".json"

    #                 print(f"Processing {raw_file}", flush=True)
    #                 data = read_json(raw_file)
    #                 process_ground_truth(data, output_file)

if __name__ == '__main__':
    main()
