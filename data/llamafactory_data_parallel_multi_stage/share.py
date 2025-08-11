MODEL_PATH = "/data1/model_init/Qwen2.5-Coder-1.5B"

"""
Raw and output data.
"""
RAW_DATA = "/data1/gsn/CustomTrain/data/88repo_1000_no_lsp/llamafactory_data_parallel_multi_stage/stage_scale_2/valid.json"
NEW_DATA = "/data1/gsn/CustomTrain/data/88repo_1000_no_lsp/llamafactory_data_parallel_multi_stage/stage_scale_2/valid_zeta.json"
BALANCED_DATA = "/data1/gsn/CustomTrain/data/88repo_1000_no_lsp/llamafactory_data_parallel_multi_stage/stage_scale_2/balance_valid_zeta.json"
TAGGED_DATA = "/data1/gsn/CustomTrain/data/llamafactory_data_parallel_multi_stage/data_postprocessing/data/tagged/debug.json"
TAG_SELECTED_DATA = "/data1/gsn/CustomTrain/data/llamafactory_data_parallel_multi_stage/data_postprocessing/data/tag_selected/debug.json"

"""
Prompt
"""
INSTRUCTION = '### Instruction:\nYou are an ArkTS code completion assistant and your task is to analyze user edits and then rewrite an excerpt that the user provides, suggesting the appropriate edits within the excerpt, taking into account the cursor location.\n\n'

"""
Markers
"""
INPUT_MARKERS = {
    'start': '### User Excerpt:\n\n',
    'reference': '### User Edits Reference:\n\n',
    'response': '### Response:\n\n'
}
EDITABLE_MARKERS = {
    'start': '<|editable_region_start|>\n',
    'end': '\n<|editable_region_end|>'
}

"""
Data balancing
"""
NEGATIVE_KEEP_PROB = 0.25 # for balancing data

"""
Data tagging
"""
TAGGING_PROMPT_TEMPLATE: str = "arkui_template.md"

# number of samples for each tag
TAG_SELECTION_THRESHOLD = {
    '文本通用': 1,
    '尺寸设置': 5
}
