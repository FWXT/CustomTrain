MODEL_PATH = "/data1/gsn/CustomTrain/output/zeta_output/1.5b_07231700"

"""
Raw and output data.
"""
RAW_DATA = "/data1/gsn/CustomTrain/data/llamafactory_data_parallel_multi_stage/data_postprocessing/data/raw/debug.json"
NEW_DATA = "/data1/gsn/CustomTrain/data/llamafactory_data_parallel_multi_stage/data_postprocessing/data/new/debug.json"
BALANCED_DATA = "/data1/gsn/CustomTrain/data/llamafactory_data_parallel_multi_stage/data_postprocessing/data/balanced/debug.json"
TAGGED_DATA = "/data1/gsn/CustomTrain/data/llamafactory_data_parallel_multi_stage/data_postprocessing/data/tagged/debug.json"
TAG_SELECTED_DATA = "/data1/gsn/CustomTrain/data/llamafactory_data_parallel_multi_stage/data_postprocessing/data/tag_selected/debug.json"

"""
Markers
"""
INPUT_MARKERS = {
    'start': '### User Excerpt:\n\n',
    'reference': '### User Edits Reference:\n\n',
    'response': '### Response:\n\n'
}
EDITABLE_MARKERS = {
    'start': '<extra_id_start>\n',
    'end': '\n<extra_id_end>'
}

"""
Data balancing
"""
NEGATIVE_KEEP_PROB = 0.25 # for balancing data

"""
Data tagging
"""
TAGGING_PROMPT_TEMPLATE = "arkui_template.md"

# number of samples for each tag
TAG_SELECTION_THRESHOLD = {
    '文本通用': 1,
    '尺寸设置': 5
}
