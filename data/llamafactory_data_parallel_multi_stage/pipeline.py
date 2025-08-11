import argparse
import asyncio
import copy
import difflib
import json
import os

import data_postprocessing.utils as utils
import share
from balance_data import balance_data
from tagging_system.tag import tag_data
from tqdm import tqdm


def get_unified_diff(text1, text2):
    diff = difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile="Original",
        tofile="Modified",
    )
    return ''.join(diff)

def raw_data_formatting(raw_file: str, new_file: str, keep_output: bool = True):
    raw_data = utils.read_json(raw_file)

    new_data = []
    for obj in tqdm(raw_data, total=len(raw_data)):
        new_obj = copy.deepcopy(obj)

        sections = utils.extract_sections_from_input(new_obj['input'], share.INPUT_MARKERS)

        # process main section
        new_editable_section = utils.add_marker_around_editable_section(sections['editable_section'], share.EDITABLE_MARKERS)
        sections['main_section'] = utils.replace_editable_section(sections['main_section'], sections['editable_section'], new_editable_section)
        new_obj['input'] = utils.concat_sections_with_markers(sections, share.INPUT_MARKERS)

        # process output
        output_code = utils.get_code_from_diff(sections['editable_section'], new_obj['output'])
        new_output_section = utils.add_marker_around_output(output_code, share.EDITABLE_MARKERS)
        new_obj['is_unchanged'] = utils.is_unchanged_output(new_obj['output'])
        if keep_output:
            new_obj['diff_ground_truth'] = new_obj['output']
        new_obj['output'] = new_output_section

        # process unidiff for tagging
        editable_code = utils.get_code_from_diff(sections['editable_section'], '')
        new_obj['editable_code'] = editable_code
        new_obj['output_code'] = output_code
        new_obj['diff'] = get_unified_diff(editable_code, output_code)

        # process prompt
        new_obj['instruction'] = share.INSTRUCTION

        new_data.append(new_obj)

    utils.write_json(new_data, new_file)

    new_jsonl_file = os.path.splitext(new_file)[0] + '.jsonl'
    utils.write_jsonl(new_data, new_jsonl_file)

def data_balancing(raw_file: str, new_file: str):
    balance_data(raw_file, new_file, negative_keep_prob=share.NEGATIVE_KEEP_PROB, shuffle=True)

def data_tagging(raw_file: str, new_file: str):
    asyncio.run(tag_data(raw_file, new_file, share.TAGGING_PROMPT_TEMPLATE))

def select_data_by_tags(raw_file: str, new_file: str, target_tags: list[str], threshold: dict[str, int]):
    data = utils.read_json(raw_file)

    # Select samples by tags
    selected_samples = {tag: [] for tag in target_tags}
    for i, obj in enumerate(data):
        for tag in target_tags:
            if tag in obj['tags']:
                selected_samples[tag].append(i)

    # Statistics: {number of samples for some tag}/{threshold[tag]}
    diff_stat = {tag: f'{len(selected_samples[tag])}/{threshold[tag]}' for tag in target_tags}
    print(f'Sample statistics: {json.dumps(diff_stat, indent=2, ensure_ascii=False)}')

    # Filtered by threshold
    selected_samples = {k: v for k, v in selected_samples.items() if len(v) >= threshold[k]}
    print(f'Filtered tags: {json.dumps(list(selected_samples.keys()), ensure_ascii=False)}')

    # Deduplication
    sample_indices = set()
    for v in selected_samples.values():
        sample_indices.update(v)
    print(f'Selected samples: {len(sample_indices)}/{len(data)}')

    if sample_indices:
        new_data = [data[i] for i in sample_indices]
        utils.write_json(new_data, new_file)

def pipeline(is_balanced: bool,
             is_tagged: bool,
             is_selecting_tags: bool, target_tags: list[str], threshold: dict[str, int]):
    if is_selecting_tags:
        print('\nSelecting data by tags...')
        select_data_by_tags(share.TAGGED_DATA, share.TAG_SELECTED_DATA, target_tags, threshold)
    else:
        print('\nFormatting raw data...')
        raw_data_formatting(share.RAW_DATA, share.NEW_DATA)

        new_file = share.NEW_DATA
        if is_balanced:
            print('\nBalancing data...')
            data_balancing(new_file, share.BALANCED_DATA)
            new_file = share.BALANCED_DATA

        if is_tagged:
            print('\nTagging data...')
            data_tagging(new_file, share.TAGGED_DATA)

    print('\nAll tasks finished')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_balanced", action="store_true", help="If balancing the data")
    parser.add_argument("--is_tagged", action="store_true", help="If tagging the data")
    parser.add_argument("--tag_selection", type=lambda s: s.split(','), help="Select samples from target tags, separated by comma")
    args = parser.parse_args()

    is_selecting_tags = args.tag_selection is not None

    pipeline(is_balanced=args.is_balanced,
             is_tagged=args.is_tagged,
             is_selecting_tags=is_selecting_tags,
             target_tags=args.tag_selection,
             threshold=share.TAG_SELECTION_THRESHOLD)
