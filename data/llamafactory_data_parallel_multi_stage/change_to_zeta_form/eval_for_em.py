from common import _Tokenizer, tokens_to_change, inline_output_tokens, normalize_code_by_ast
import json
import pandas as pd
from typing import List, Dict, Any
import os

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f"{PARENT_DIR}/data/v2/2"

EVAL_MODEL = "0.5b_ckpt17000"
EVAL_RESULT_DIR = f"{PARENT_DIR}/eval_result/v2/2/{EVAL_MODEL}"
DATA_FILE = f"{DATA_DIR}/random_5000_test_v2_output_{EVAL_MODEL}.json"

JSONL_OUTPUT_FILE = f"{EVAL_RESULT_DIR}/result_list.jsonl"
EXCEL_OUTPUT_FILE = f"{EVAL_RESULT_DIR}/evaluation_report.xlsx"


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
    
    if start_pos == -1 or end_pos == -1:
        print("警告: 未找到指定的标记")
        return input_text
    
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

def get_code(input, ground_truth, output):
    """
    将输入文本转换为token_ids，然后计算代码变化
    Args:
        input: 输入文本
        ground_truth: 真实答案
        output: 模型输出
    Returns:
        tuple: (pred_code, label_code) 预测代码和标签代码
    """
    # 使用tokenizer将文本转换为token_ids
    input_tokens = _Tokenizer.encode(input, add_special_tokens=False)
    ground_truth_tokens = _Tokenizer.encode(ground_truth, add_special_tokens=False)
    output_tokens = _Tokenizer.encode(output, add_special_tokens=False)
    
    pred_code = tokens_to_change(
        inline_output_tokens(input_tokens, output_tokens)
    ).after
    label_code = tokens_to_change(
        inline_output_tokens(input_tokens, ground_truth_tokens)
    ).after
    return pred_code, label_code

def read_output(output_path):
    """
    读取JSON文件并返回数据列表
    Args:
        output_path: JSON文件路径
    Returns:
        list: JSON数据列表
    """
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        print(f"成功读取 {len(data_list)} 条数据")
        return data_list
    except FileNotFoundError:
        print(f"文件未找到: {output_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return []
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []

def code_equal(code1: str, code2: str) -> bool:
    if code1 == code2:
        return True
    code1 = normalize_code_by_ast(code1)
    code2 = normalize_code_by_ast(code2)
    return code1 == code2

def eval_em(full_input, ground_truth, output):
    # 提取main_input部分(带<extra_id_x>标签)
    main_input = extract_main_input(full_input)
    pred_code, label_code = get_code(main_input, ground_truth, output)
    print(f"main_input: \n{main_input}")
    print(f"pred_code: \n{pred_code}")
    print(f"label_code: \n{label_code}\n\n")
    # 计算精确匹配
    return code_equal(pred_code, label_code), main_input, pred_code, label_code

def extract_diff_content(text, main_input):
    """
    从文本中提取diff内容
    Args:
        text: 包含diff信息的文本 (ground_truth 或 output)
        main_input: 原始输入内容，用于获取<extra_id_x>对应的原文
    Returns:
        list: diff内容列表，格式为 ["行号 - 删除内容;行号 + 新增内容", ...]
    """
    import re
    
    # 提取main_input中<extra_id_x>对应的原文
    main_lines = main_input.split('\n')
    extra_id_to_content = {}
    
    for line in main_lines:
        # 匹配<extra_id_x>格式
        match = re.match(r'<extra_id_(\d+)>(.*)', line)
        if match:
            line_num = int(match.group(1))
            content = match.group(2).strip()
            extra_id_to_content[line_num] = content
    
    # 按<extra_id_x>分割文本
    extra_id_pattern = r'<extra_id_\d+>'
    segments = re.split(extra_id_pattern, text)
    extra_id_matches = re.findall(extra_id_pattern, text)
    
    diff_list = []
    
    # 处理每个分割段
    for i, (segment, extra_id_match) in enumerate(zip(segments[1:], extra_id_matches)):  # 跳过第一个空段
        # 提取行号
        line_num = int(extra_id_match.replace('<extra_id_', '').replace('>', ''))
        
        # 清理segment，移除<s>和</s>标记
        segment = segment.replace('<s>', '').replace('</s>', '').strip()
        
        if not segment:  # 空段，跳过
            continue
            
        # 检查是否包含<add>和<del>标记
        if '<add>' in segment and '<del>' in segment:
            # 提取所有add内容
            add_matches = re.findall(r'<add>(.*?)(?=<add>|<del>|$)', segment, re.DOTALL)
            if add_matches:
                # 合并所有add内容，用换行符分隔，并对每行进行strip
                add_content = '\n'.join([content.strip() for content in add_matches])
                del_content = extra_id_to_content.get(line_num, "").strip()
                
                diff_entry = f"{line_num} - {del_content};{line_num} + {add_content}"
                diff_list.append(diff_entry)
                
        elif '<add>' in segment:
            # 只有add标记，可能有多个
            add_matches = re.findall(r'<add>(.*?)(?=<add>|$)', segment, re.DOTALL)
            if add_matches:
                # 合并所有add内容，用空格分隔，并对每行进行strip
                add_content = ' '.join([content.strip() for content in add_matches])
                
                # 只有add标记时，输出格式为 "行号 + 内容"
                diff_entry = f"{line_num} + {add_content}"
                diff_list.append(diff_entry)
                
        elif '<del>' in segment:
            # 只有del标记
            del_match = re.search(r'<del>(.*)', segment, re.DOTALL)
            if del_match:
                del_content = del_match.group(1).strip()
                original_content = extra_id_to_content.get(line_num, "").strip()
                
                # 只有del标记时，输出格式为 "行数 - 行内容"
                diff_entry = f"{line_num} - {original_content}"
                diff_list.append(diff_entry)
    
    return diff_list

def eval_diff_line_gain(full_input, ground_truth, output):
    """
    评估ground_truth和output的diff内容差异
    Args:
        full_input: 完整输入
        ground_truth: 真实答案
        output: 模型输出
    Returns:
        dict: 包含diff比较结果的字典
    """
    # 提取main_input
    main_input = extract_main_input(full_input)
    
    # 提取ground_truth和output的diff内容
    gt_diff = extract_diff_content(ground_truth, main_input)
    output_diff = extract_diff_content(output, main_input)
    
    # 计算diff内容的匹配情况
    gt_set = set(gt_diff)
    output_set = set(output_diff)
    
    # 计算TP（真正例）：预测对了且需要改动的行
    tp = len(gt_set & output_set)  # 交集：正确预测的改动
    
    # 计算FP（假正例）：预测了但是不应该改动行
    fp = len(output_set - gt_set)  # 差集：错误预测的改动
    
    # 计算FN（假负例）：应该改动但没有预测的行
    fn = len(gt_set - output_set)  # 差集：漏掉的改动
    
    total_gt = len(gt_set)  # ground truth总数
    total_output = len(output_set)  # output总数
    
    precision = tp / total_output if total_output > 0 else 0
    recall = tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算diff-line gain指标：TP - FP
    diff_line_gain = tp - fp
    
    result = {
        "gt_diff": gt_diff,
        "output_diff": output_diff,
        "tp": tp,  # 真正例
        "fp": fp,  # 假正例
        "fn": fn,  # 假负例
        "total_gt": total_gt,
        "total_output": total_output,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "diff_line_gain": diff_line_gain  # diff-line gain指标
    }
    
    return result

def is_unchanged_data(s: str) -> bool:
    return (not "<add>" in s) and (not "<del>" in s)

def eval(data_file: str, output_file: str = "result_list.jsonl"):
    """
    评估函数，计算精确匹配率
    """
    data_list = read_output(data_file)
    if not data_list:
        print("没有读取到数据")
        return
    
    correct_count = 0
    total_count = len(data_list)
    result_list = []
    indices = {"correct": [], "wrong": [], "ground_truth_unchanged": [], "output_unchanged": []}
    
    for i, data in enumerate(data_list):
        try:
            record = {}
            full_input = data["input"]
            ground_truth = data["ground_truth"]
            output = data["output"]
            record["full_input"] = data["input"]
            record["ground_truth"] = ground_truth
            record["output"] = output
            record["em_result"], record["main_input"], record["pred_code"], record["label_code"] = eval_em(full_input, ground_truth, output)
            record["diff_result"] = eval_diff_line_gain(full_input, ground_truth, output)
            if record["em_result"]:
                correct_count += 1
                indices["correct"].append(i)
            else:
                indices["wrong"].append(i)

            if is_unchanged_data(ground_truth):
                indices["ground_truth_unchanged"].append(i)
            if is_unchanged_data(output):
                indices["output_unchanged"].append(i)

            result_list.append(record)
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{total_count} 条数据")
        except Exception as e:
            print(f"处理第 {i+1} 条数据时出错: {e}")
            continue
    with open(output_file, "w", encoding="utf-8") as f:
        for record in result_list:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(f"{EVAL_RESULT_DIR}/indices.json", "w", encoding="utf-8") as f:
        json.dump(indices, f)
    
    # 计算精确匹配率
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\n评估结果:")
    print(f"总数据量: {total_count}")
    print(f"正确数量: {correct_count}")
    print(f"精确匹配率: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print(f"unchanged ground truth: {len(indices['ground_truth_unchanged'])} / {total_count} ({len(indices['ground_truth_unchanged']) / total_count * 100:.2f}%)")
    print(f"unchanged output: {len(indices['output_unchanged'])} / {total_count} ({len(indices['output_unchanged']) / total_count * 100:.2f}%)")
    
    return result_list

def output_report(result_list: List[Dict[str, Any]], output_file: str = "evaluation_report.xlsx"):
    """
    将评估结果输出到Excel文件
    Args:
        result_list: 评估结果列表
        output_file: 输出文件名
    """
    if not result_list:
        print("没有数据可以输出")
        return
    
    # 创建Excel写入器
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # 准备summary数据
        summary_data = {}
        
        # 计算各项指标的平均值
        total_cases = len(result_list)
        em_correct = sum(1 for record in result_list if record.get("em_result", False))
        
        # EM指标
        summary_data["总案例数"] = total_cases
        summary_data["EM正确数"] = em_correct
        summary_data["EM准确率"] = em_correct / total_cases if total_cases > 0 else 0
        
        # Diff指标的总数和平均值
        diff_metrics = ["tp", "fp", "fn", "precision", "recall", "f1", "diff_line_gain"]
        for metric in diff_metrics:
            values = [record.get("diff_result", {}).get(metric, 0) for record in result_list]
            total_value = sum(values) if values else 0
            avg_value = total_value / len(values) if values else 0
            
            # 对于tp、fp、fn、diff_line_gain，同时显示总数和平均值
            if metric in ["tp", "fp", "fn", "diff_line_gain"]:
                summary_data[f"总{metric.upper()}"] = total_value
                summary_data[f"平均{metric.upper()}"] = avg_value
            else:
                # 对于precision、recall、f1，只显示平均值
                summary_data[f"平均{metric.upper()}"] = avg_value
        
        # 创建summary DataFrame
        summary_df = pd.DataFrame(list(summary_data.items()), columns=["指标", "值"])
        
        # 写入summary sheet
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        
        # 准备detail数据
        detail_data = []
        
        for i, record in enumerate(result_list):
            row_data = {
                "案例ID": i + 1,
                "EM结果": record.get("em_result", False),
                "Ground Truth": record.get("ground_truth", ""),
                "模型输出": record.get("output", ""),
                "Main Input": record.get("main_input", ""),
                "预测代码": record.get("pred_code", ""),
                "标签代码": record.get("label_code", "")
            }
            
            # 添加diff_result中的各项指标
            diff_result = record.get("diff_result", {})
            for key, value in diff_result.items():
                if key not in ["gt_diff", "output_diff"]:  # 跳过列表类型的字段
                    row_data[f"diff_{key}"] = value
            
            # 添加diff内容（转换为字符串）
            row_data["GT_Diff"] = "; ".join(diff_result.get("gt_diff", []))
            row_data["Output_Diff"] = "; ".join(diff_result.get("output_diff", []))
            
            detail_data.append(row_data)
        
        # 创建detail DataFrame
        detail_df = pd.DataFrame(detail_data)
        
        # 写入detail sheet
        detail_df.to_excel(writer, sheet_name="detail", index=False)
    
    print(f"评估报告已保存到: {output_file}")
    print(f"Summary sheet包含 {len(summary_data)} 个指标")
    print(f"Detail sheet包含 {len(detail_data)} 个案例")

if __name__ == "__main__":
    result_list = eval(DATA_FILE, JSONL_OUTPUT_FILE)
    output_report(result_list, output_file=EXCEL_OUTPUT_FILE)
