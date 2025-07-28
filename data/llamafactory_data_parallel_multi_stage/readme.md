# Multi-Scale Data
论文分为多scale，代表reference token数目，越大用的token越少。并且scale 2会包含scale 4的数据，学习逐渐过渡由短到长。所以先scale 4训练。
## 基础data用各个scale下的train.json, valid.json, test.json
## 需要针对各个技术方案对上述json的字段进行修改使得效果好+模型小

# Data Postprocessing
```shell
LLM_API_KEY=YOUR-API-KEY python pipeline.py --is_balanced --is_tagged
```

pipeline包含以下流程：
1. raw_data_formatting: 将原始数据转换为统一格式，原始数据格式为coeditor输入输出格式
2. data_balancing: 对数据正负样本进行平衡处理（负样本：no-op），使用`--is_balanced`启用，默认不启用
3. data_tagging: 访问LLM API，对数据进行打标处理，使用`--is_tagged`启用，默认不启用

pipeline中涉及参数可通过share.py进行修改

# Selecting Data by Tags
```shell
python pipeline.py --tag_selection 文本通用,尺寸设置
```

`--tag_selection`与data postprocessing流程互斥，若启用`--tag_selection`，则只会从share.py中的`TAGGED_DATA`中读取数据，写入到`TAG_SELECTED_DATA`中，建议执行data postprocessing流程后，再执行该流程

标签列表用英文逗号分隔，阈值为share.py中的`TAG_SELECTION_THRESHOLD`，表示不同标签至少需要的样本数
