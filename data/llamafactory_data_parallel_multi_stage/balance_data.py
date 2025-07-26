import json
import random
# 原始文件路径
input_file = "/data/mnt_bucket/qzq/CustomTrain/data/llamafactory_data_parallel_multi_stage/stage_scale_2/necessary_train_zeta.json"
# 输出文件路径
output_file = "/data/mnt_bucket/qzq/CustomTrain/data/llamafactory_data_parallel_multi_stage/stage_scale_2/balanced_necessary_train_zeta.json"
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

def balance_data(input_file, output_file, negative_keep_prob:float=0.333):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

        sample_data = []
        for item in data:
            if item["is_unchanged"]:
                # random prob less than 1/2, keep it
                if random.random() < negative_keep_prob:
                    sample_data.append(item)
            else:
                sample_data.append(item)

        data = sample_data

    # 保存到新文件，确保中文不被转义
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    stat = inpsect_unchange_ratio("/data1/public/CustomTrain/data/llamafactory_data_parallel_multi_stage/stage_scale_2/train_zeta.json")
    print(stat)
