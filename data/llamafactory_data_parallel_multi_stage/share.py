MODEL_PATH = "/data1/gsn/CustomTrain/output/zeta_output/1.5b_07231700"

"""
Raw and output data.
"""
RAW_DATA = "/data1/gsn/CustomTrain/data/llamafactory_data_parallel_multi_stage/data_postprocessing/data/raw/debug.json"
NEW_DATA = "/data1/gsn/CustomTrain/data/llamafactory_data_parallel_multi_stage/data_postprocessing/data/output/debug.json"

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
