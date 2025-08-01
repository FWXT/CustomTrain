import json
import os
import re
from collections import Counter

from utils import write_json


# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)

def contains_more_than_threshold_chinese_chars(text: str, threshold: int = 3) -> bool:
    if text == "":
        return True # discard empty string samples
    chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
    return len(chinese_chars) > threshold

def contains_copyright(text: str) -> bool:
    if text == "":
        return True # discard empty string samples
    return "copyright" in text.lower() or "license" in text.lower()

def contains_too_much_prefix(text: str, prefix_char_num: int = 7) -> bool:
    extra_id_pattern = r'<extra_id_\d+>'
    clean_text = re.sub(extra_id_pattern, '', text)

    lines = clean_text.split('\n')
    prefix_counts = Counter()
    for line in lines:
        tmp = line.strip()
        if len(tmp) > prefix_char_num:
            prefix = tmp[:10]
            prefix_counts[prefix] += 1
    if all(count <= 6 for count in prefix_counts.values()):
        return False
    return True

def contains_reference_high_ratio_modify(reference_text: str, high_ratio: float = 0.9) -> bool:
    lines_num = reference_text.count("\n")
    add_num = reference_text.count("<add>")
    delete_num = reference_text.count("<del>")
    modify_num = add_num + delete_num
    modify_ratio = 1.0 * modify_num / lines_num
    flag =  modify_ratio >= high_ratio and lines_num > 4
    return flag

def contains_reference_low_ratio_modify(reference_text: str, low_ratio: float = 0.05) -> bool:
    lines_num = reference_text.count("\n")
    add_num = reference_text.count("<add>")
    delete_num = reference_text.count("<del>")
    modify_num = add_num + delete_num
    modify_ratio = 1.0 * modify_num / lines_num
    flag =  modify_ratio <= low_ratio and modify_num > 0 # 避开unchange的
    return flag

def references_bad(references: list[dict[str, str]]) -> bool:
    flattened_references = [value for d in references for value in d.values()]

    if any(contains_reference_high_ratio_modify(reference, 0.9) for reference in flattened_references):
        return True
    if any(contains_reference_low_ratio_modify(reference, 0.04) for reference in flattened_references):
        return True

    return False

def handle_file(input_file, output_file):
    with open(input_file, encoding="utf-8") as file:
        data = json.load(file)
        no_chineses = [item for item in data if not contains_more_than_threshold_chinese_chars(item["input"], 0)]
        no_copyright_chinese = [item for item in no_chineses if not contains_copyright(item["input"])]
        no_copyright_chinese_no_share_prefix = [item for item in no_copyright_chinese if not contains_too_much_prefix(item["input"], 7)]

        phase1_target = no_copyright_chinese_no_share_prefix

        no_high_ratio_modify_reference = [item for item in phase1_target if not references_bad(item["references"])]

    chinese_stat = {
        "all": len(data),
        "no_chinese": len(no_chineses),
        "no_chinese_copyright": len(no_copyright_chinese),
        "no_copyright_chinese_no_share_prefix": len(no_copyright_chinese_no_share_prefix),
        "final_consider_modify_reference_ratio": len(no_high_ratio_modify_reference)
    }
    print("chinese_stat", chinese_stat)

    write_json(no_high_ratio_modify_reference, output_file)

    print()

def main():
    # stage = "stage_scale_2"
    # train_file = os.path.join(current_dir, stage, "balanced_train_zeta.json")
    # valid_file = os.path.join(current_dir, stage, "balanced_valid_zeta.json")
    # test_file = os.path.join(current_dir, stage, "balanced_test_zeta.json")

    # new_train_file = os.path.join(current_dir, stage, "filter_balanced_train_zeta.json")
    # new_valid_file = os.path.join(current_dir, stage, "filter_balanced_valid_zeta.json")
    # new_test_file = os.path.join(current_dir, stage, "filter_balanced_test_zeta.json")

    # handle_file(train_file, new_train_file)
    # handle_file(valid_file, new_valid_file)


    # stage = "stage_scale_4"
    # train_file = os.path.join(current_dir, stage, "balanced_train_zeta.json")
    # valid_file = os.path.join(current_dir, stage, "balanced_valid_zeta.json")
    # test_file = os.path.join(current_dir, stage, "balanced_test_zeta.json")

    # new_train_file = os.path.join(current_dir, stage, "filter_balanced_train_zeta.json")
    # new_valid_file = os.path.join(current_dir, stage, "filter_balanced_valid_zeta.json")
    # new_test_file = os.path.join(current_dir, stage, "filter_balanced_test_zeta.json")

    # handle_file(train_file, new_train_file)
    # handle_file(valid_file, new_valid_file)

    raw_file = "/data1/gsn/CustomTrain/data/88repo_100_no_lsp_/llamafactory_data_parallel_multi_stage/stage_scale_4/balance_train_zeta.json"
    new_file = "/data1/gsn/CustomTrain/data/88repo_100_no_lsp_/llamafactory_data_parallel_multi_stage/stage_scale_4/filter_balance_train_zeta.json"

    handle_file(raw_file, new_file)

if __name__ == "__main__":
    main()
