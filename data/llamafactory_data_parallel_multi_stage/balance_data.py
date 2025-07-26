import json
import random
import os
random.seed(42)
# 读取 JSON 数据

def inpsect_unchange_ratio(input_file):
    stat = {}
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        count = 0
        unchange_count = 0
        for item in data:
            count+=1
            if item["is_unchanged"]:
                unchange_count += 1
    stat["all"] = count
    stat["is_unchanged"] = unchange_count
    print("unchange ratio:", 1.0*unchange_count/count)
    return stat

def balance_data(input_file, output_file, negative_keep_prob:float=0.333, shuffle:bool=True):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        positive_num = 0
        before_negative_num = 0
        after_negative_num = 0
        sample_data = []
        for item in data:
            if item["is_unchanged"]:
                before_negative_num += 1
                if random.random() < negative_keep_prob:
                    sample_data.append(item)
                    after_negative_num += 1
            else:
                sample_data.append(item)
                positive_num += 1

        data = sample_data
    
    # before change
    before_all_num = before_negative_num+positive_num
    is_unchanged_ratio = 1.0*before_negative_num/before_all_num
    before_stat = {"all":before_all_num,"is_unchanged":before_negative_num,"is_unchanged_ratio":is_unchanged_ratio}
    # after change
    after_all_num = after_negative_num+positive_num
    is_unchanged_ratio = 1.0*after_negative_num/after_all_num
    after_stat = {"all":after_all_num,"is_unchanged":after_negative_num,"is_unchanged_ratio":is_unchanged_ratio}
    # 保存到新文件，确保中文不被转义

    print("before balance stat:",before_stat)
    print("after balance stat:",after_stat)
    if shuffle:
        random.shuffle(sample_data)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)


def balance_folder(folder_path, negative_keep_prob=0.25, shuffle=True):
    """
    Process all JSON files containing 'zeta' in the given folder,
    balance them and save with 'balanced_' prefix.
    
    Args:
        folder_path: Path to the folder containing JSON files
        negative_keep_prob: Probability to keep negative samples
        shuffle: Whether to shuffle the balanced data
    """
    # List all files in the folder
    for filename in os.listdir(folder_path):
        # Check if file is JSON and contains 'zeta'
        if filename.endswith('.json') and 'zeta' in filename.lower():
            input_file = os.path.join(folder_path, filename)
            output_file = os.path.join(folder_path, f"balanced_{filename}")
            
            print(f"\nProcessing file: {input_file}")
            print(f"Output will be saved to: {output_file}")
            
            # Apply balance_data function
            balance_data(input_file, output_file, negative_keep_prob, shuffle)


if __name__ == "__main__":
    # stat = inpsect_unchange_ratio("/data1/public/CustomTrain/data/llamafactory_data_parallel_multi_stage/stage_scale_1/train_zeta.json")
    # print(stat)

    # # 原始文件路径
    # input_file = "/data1/public/CustomTrain/data/llamafactory_data_parallel_multi_stage/stage_scale_2/test_zeta.json"
    # # 输出文件路径
    # output_file = "/data1/public/CustomTrain/data/llamafactory_data_parallel_multi_stage/stage_scale_2/balanced_test_zeta.json"

    # balance_data(input_file, output_file, negative_keep_prob=0.25, shuffle=True)
    negative_keep_prob=0.25
    shuffle=True

    target_folder = "/data1/public/CustomTrain/data/llamafactory_data_parallel_multi_stage/stage_scale_1"
    balance_folder(target_folder,negative_keep_prob=negative_keep_prob,shuffle=shuffle)

    target_folder = "/data1/public/CustomTrain/data/llamafactory_data_parallel_multi_stage/stage_scale_2"
    balance_folder(target_folder,negative_keep_prob=negative_keep_prob,shuffle=shuffle)

    target_folder = "/data1/public/CustomTrain/data/llamafactory_data_parallel_multi_stage/stage_scale_4"
    balance_folder(target_folder,negative_keep_prob=negative_keep_prob,shuffle=shuffle)



