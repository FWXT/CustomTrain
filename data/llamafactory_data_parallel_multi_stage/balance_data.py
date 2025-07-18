import json
import random
# 原始文件路径
input_file = "/data/mnt_bucket/qzq/CustomTrain/data/llamafactory_data_parallel_multi_stage/stage_scale_2/necessary_train_zeta.json"
# 输出文件路径
output_file = "/data/mnt_bucket/qzq/CustomTrain/data/llamafactory_data_parallel_multi_stage/stage_scale_2/balanced_necessary_train_zeta.json"
random.seed(42)
# 读取 JSON 数据
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

    sample_data = []
    for item in data:
        if item["is_unchanged"]:
            # random prob less than 1/2, keep it
            if random.random() < 0.333:
                sample_data.append(item)
        else:
            sample_data.append(item)

    data = sample_data


# 取前 800 个样本

# 保存到新文件，确保中文不被转义
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(sample_data, f, ensure_ascii=False, indent=2)

