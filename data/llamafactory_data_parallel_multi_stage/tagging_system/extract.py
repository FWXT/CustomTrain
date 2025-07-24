import json
from typing import cast, List, Dict
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import sys
# 获取当前文件的目录和父目录
current_dir = Path(__file__).parent  # 当前文件所在目录
parent_dir = current_dir.parent      # 当前文件的父目录
sys.path.append(str(parent_dir))
data_dir = parent_dir.parent

from share import MODEL_PATH

RAW_FILE = Path(data_dir) / "coeditor-python" / "raw" / "stage_scale_4" / "test.json"
NEW_FILE = Path(current_dir) /  "test_extract.json"

MAX_REF_TOKENS = 2048

MAX_SAMPLES = 500

import difflib

def get_unified_diff(text1, text2):
    diff = difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile="Original",
        tofile="Modified",
    )
    return ''.join(diff)


# 定义断言函数
def assert_eq(a, b):
    assert a == b, f"{a} != {b}"

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, output_file: str):
    """
    将字典数据写入JSON文件，支持中文字符。

    Args:
        data (dict): 要写入的字典数据。
        output_file (str): 输出文件的路径。
    """
    with open(output_file, 'w', encoding='utf-8') as f: # 确保以 UTF-8 编码写入文件
        json.dump(data, f, indent=2, ensure_ascii=False) # <--- 关键修改: ensure


def read_jsonl(json_file: str):
    ret_list = []
    with open(json_file, "r") as f:
        try:
            for i, line in enumerate(f.readlines()):
                ret_list.append(json.loads(line))
        except:
            print(i)
    return ret_list

def load_tokenizer(local_path: str):
    try:
        # 使用本地已下载的qwen2.5-coder模型
        if Path(local_path).exists():
            print(f"✅ 使用本地qwen2.5-coder模型: {local_path}")
            tokenizer = cast(
                AutoTokenizer, AutoTokenizer.from_pretrained(
                    local_path,
                    local_files_only=True,  # 只使用本地文件
                    trust_remote_code=True
                )
            )
            print("✅ 成功加载本地 qwen2.5-coder tokenizer")

            return tokenizer
        else:
            raise FileNotFoundError(f"本地模型路径不存在: {local_path}")
            
    except Exception as e:
        print(f"⚠️  无法加载本地qwen2.5-coder tokenizer: {e}")

_Tokenizer: AutoTokenizer = load_tokenizer(MODEL_PATH)
_BaseTokenizer = _Tokenizer
def str_to_token_num(s: str | None) -> int:
    if s is None:
        return 0
    return len(_Tokenizer.encode(s, add_special_tokens=False))

def cutoff_str_by_token_num(s: str, max_len: int) -> str:
    seq = _Tokenizer.encode(s, add_special_tokens=False)[:max_len]
    output = _Tokenizer.decode(seq, add_special_tokens=False)
    # print(f"output length: {len(output)}, raw length: {len(s)}")
    # assert_eq(_Tokenizer.decode(seq, add_special_tokens=False), s)
    return output

def extract_snippets_from_input(input_text: str) -> Dict:
    start_marker = "### User Excerpt:"
    ref_marker = "### User Edits Reference:"
    response_marker = "### Response:"

    start_pos = input_text.find(start_marker)
    ref_pos = input_text.find(ref_marker)
    response_pos = input_text.find(response_marker)

    if ref_pos != -1:
        main_section = input_text[start_pos + len(start_marker):ref_pos]
    else:
        main_section = input_text[start_pos + len(start_marker):response_pos]
    
    ref_section = None
    if ref_pos != -1:
        ref_section = input_text[ref_pos + len(ref_marker):response_pos]
        ref_section = ref_section[:-2] # remove the last two '\n'
    
    return {
        "main_section": main_section,
        "ref_section": ref_section
    }


def concat_snippets(snippets: Dict) -> str:
    start_marker = "### User Excerpt:"
    ref_marker = "### User Edits Reference:"
    response_marker = "### Response:\n\n"

    output = start_marker + snippets["main_section"] + ref_marker + snippets["ref_section"]
    output += "\n\n" + response_marker    

    return output

def process_main_section(main_section: str) -> tuple[str, str]:
    import re
    extra_id_pattern = r'<extra_id_\d+>'
    region_start_marker = "<extra_id_start>"
    region_end_marker = "<extra_id_end>"
    
    main_section_lines = main_section.split("\n")
    
    # Track positions of extra_id lines (original indices)
    extra_id_indices = []
    
    for i, line in enumerate(main_section_lines):
        if re.search(extra_id_pattern, line):
            extra_id_indices.append(i)
    
    if not extra_id_indices:
        print("警告: 未找到<extra_id_标签")
        return main_section, ""
    
    # Process start marker
    first_extra_id_line = extra_id_indices[0]
    # Insert start_marker above the first extra_id line
    main_section_lines.insert(first_extra_id_line, region_start_marker)
    # We only need to adjust the first_extra_id_line for end marker processing
    first_extra_id_line += 1  # because we inserted a line before it
    last_extra_id_line = extra_id_indices[-1] + 1  # because we inserted one line before first extra_id
    
    # Process end marker
    # Find the next line that doesn't have <add> or <del>
    insert_position = None
    for i in range(last_extra_id_line + 1, len(main_section_lines)):
        if "<add>" not in main_section_lines[i] and "<del>" not in main_section_lines[i]:
            insert_position = i
            break
    
    if insert_position is None:
        # If all remaining lines have <add> or <del>, append at end
        insert_position = len(main_section_lines)
    
    # Insert end_marker at the found position
    main_section_lines.insert(insert_position, region_end_marker)
    
    # Extract the region content
    start_index = first_extra_id_line
    end_index = insert_position  # because we inserted end marker after this position
    
    # Remove all <extra_id_\d+> tags
    for i in range(len(main_section_lines)):
        main_section_lines[i] = re.sub(extra_id_pattern, '', main_section_lines[i])
    
    processed_main_section = "\n".join(main_section_lines)

    region_content = "\n".join(main_section_lines[start_index:end_index])

    return processed_main_section, region_content


def compute_percentiles(t):
    p20 = np.percentile(t, 20)
    p50 = np.percentile(t, 50)
    p90 = np.percentile(t, 90)
    return p20, p50, p90

def main():
    read_file = RAW_FILE
    write_file = NEW_FILE
    from change_to_zeta_form.process_ground_truth import get_code, extract_main_input
    data = read_json(read_file)[:MAX_SAMPLES]
    new_data = []
    for obj in tqdm(data, total=len(data)):
        new_obj = {}
        snippets = extract_snippets_from_input(obj["input"])

        if snippets["ref_section"] is not None:
            snippets["ref_section"] = cutoff_str_by_token_num(snippets["ref_section"], MAX_REF_TOKENS)
            obj["input"] = concat_snippets(snippets)
        
        main_section = snippets["main_section"]
        handled_main_section, region_content = process_main_section(main_section)
        new_obj["main_section"] = handled_main_section 

        # 复用另外文件的逻辑
        main_input = extract_main_input(obj['input'])
        output = get_code(main_input, obj['output'])
        new_obj["output"] = output
        new_obj["region_input"] = region_content

        diff = get_unified_diff(region_content, output)
        new_obj["diff"] = diff
        new_data.append(new_obj) # ? 我怎么感觉数据处理换行有问题

    write_json(new_data, write_file)

   
if __name__ == '__main__':
    main()
