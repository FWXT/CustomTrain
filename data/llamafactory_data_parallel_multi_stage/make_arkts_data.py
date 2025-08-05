import copy
import difflib
import random
import re

import data_postprocessing.utils as utils
import share
from tqdm import tqdm


RAW_FILE = "/data1/gsn/CustomTrain/data/llamafactory_data_parallel_multi_stage/example.json"
NEW_FILE = "/data1/gsn/CustomTrain/data/llamafactory_data_parallel_multi_stage/example_new.json"

def extract_variable_names_from_line_change(line_change: str) -> list[tuple[str, str]]:
    """Extract varialbe names decorated by @State from line change."""
    pattern = r'^\s*(?:@\w+\s*)+\s*([a-zA-Z_]\w*)[\s:=]'

    names = []
    changes = line_change.split('\n')
    has_del = '<del>' in line_change

    for i, change in enumerate(changes):
        if change.startswith(' <add> ') and '@State' in change:
            change_code = change.replace(' <add> ', '')
            m = re.search(pattern, change_code)
            if m:
                name = m.group(1)

                copy_changes = copy.deepcopy(changes)
                copy_changes.pop(i)
                new_line_change = '\n'.join(copy_changes)
                if not has_del:
                    new_line_change += '\n'

                names.append((name, new_line_change))
    return names

def get_name_and_usage_pairs_by_line(
        line_changes: list[str],
        names_by_line: list[list[tuple[str, str]]]) -> list[tuple[int, int, str, str]]:
    def is_matched(name: str, line_change: str) -> bool:
        pattern = 'this.' + name
        return pattern in line_change

    assert len(line_changes) == len(names_by_line), 'line_changes and names_by_line should have the same length'

    n, pairs = len(line_changes), []
    for i, names in enumerate(names_by_line):
        for name, new_line_change in names:
            for j in range(i + 1, n):
                if is_matched(name, line_changes[j]):
                    pairs.append((i, j, name, new_line_change))

    return pairs

def process_usage_line_change(line_change: str, name: str):
    def random_mask(change: str, keep_ratio: float = 0.5) -> str:
        prefix = ' <add> '
        suffix = change[len(prefix):]
        split_index = random.randint(int(len(suffix) * keep_ratio), len(suffix) - 1)
        return prefix + suffix[:split_index]

    pattern = 'this.' + name
    changes = line_change.split('\n')
    has_del = '<del>' in line_change

    cases = []
    for i, change in enumerate(changes):
        if pattern in change:
            copy_changes = copy.deepcopy(changes)
            masked_change = random_mask(change) + '<|user_cursor_is_here|>'
            copy_changes[i] = masked_change
            case = '\n'.join(copy_changes)
            if not has_del:
                case += '\n'
            cases.append(case)

    return cases

def make_diff_from_new_line_change(raw_line_changes: list[str], new_line_change: str, line_id: int) -> str:
    new_line_changes = copy.deepcopy(raw_line_changes)
    new_line_changes[line_id] = new_line_change

    # Discard all changes after line_id
    for i in range(line_id + 1, len(new_line_changes)):
        new_line_changes[i] = ''

    new_line_changes = [f'<extra_id_{i}>' + new_line_changes[i] for i in range(len(new_line_changes))]
    diff = ''.join(new_line_changes)
    return '<s>' + diff + '</s>'

def get_unidiff(x: str, y: str,
                fromfile: str = '', tofile: str = '',
                remove_header: bool = False) -> str:
    unidiff = list(difflib.unified_diff(
        x.splitlines(keepends=True),
        y.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=tofile
    ))
    if remove_header:
        unidiff = unidiff[2:]
    return ''.join(unidiff)

def make_event_text(sections: dict[str, str], before_event_diff: str, after_event_diff: str) -> str:
    def extract_editting_filename(main_section: str) -> str:
        newline_pos = main_section.find('\n') # filename is in the first line
        return main_section[:newline_pos][len('# module: '):]

    filename = extract_editting_filename(sections['main_section'])
    header = f'### User Edits:\n\nUser edited file: \"{filename}\":\n\n'

    editable_section = sections['editable_section']
    old_editable_code = utils.get_code_from_diff(editable_section, before_event_diff)
    new_editable_code = utils.get_code_from_diff(editable_section, after_event_diff)

    # Discard '---' and '+++' lines
    unidiff = get_unidiff(old_editable_code, new_editable_code, remove_header=True)
    return header + '```diff\n' + unidiff + '```'

def process(input_text: str, output_text: str, no_ref: bool = False):
    extra_id_pattern = r'<extra_id_\d+>'

    raw_sections = utils.extract_sections_from_input(input_text)

    raw_editable_section = raw_sections['editable_section']
    clean_output = output_text.replace('<s>', '').replace('</s>', '')

    # Remove the first empty segment which is before <extra_id_0>
    line_changes = re.split(extra_id_pattern, clean_output)[1:]

    names_by_line = [extract_variable_names_from_line_change(line_change) for line_change in line_changes]

    pairs = get_name_and_usage_pairs_by_line(line_changes, names_by_line)

    result = []
    for pair in pairs:
        name_line_id, usage_line_id, name, event_line_change = pair

        # make new diffs for `input`
        cases = process_usage_line_change(line_changes[usage_line_id], name)
        # TODO: if discard all changes after current extra id?
        input_diffs = [make_diff_from_new_line_change(line_changes, case, usage_line_id) for case in cases]
        output_diff = make_diff_from_new_line_change(line_changes, line_changes[usage_line_id], usage_line_id)

        # make event text
        before_event_diff = make_diff_from_new_line_change(line_changes, event_line_change, name_line_id) # before adding variable definition
        after_event_diff = make_diff_from_new_line_change(line_changes, line_changes[name_line_id], name_line_id) # after adding variable definition
        event_text = make_event_text(raw_sections, before_event_diff, after_event_diff)

        for diff in input_diffs:
            sections = copy.deepcopy(raw_sections)

            new_editable_section = utils.get_code_from_diff(raw_editable_section, diff)
            new_editable_section = utils.add_marker_around_editable_section(new_editable_section, markers=share.EDITABLE_MARKERS)

            # process new input
            sections['main_section'] = utils.replace_editable_section(sections['main_section'], raw_editable_section, new_editable_section)
            if no_ref:
                sections['reference_section'] = ''
            sections['main_section'] = sections['main_section'].replace('# module: ', '```').rstrip() + '\n```\n\n'
            new_input_text = utils.concat_sections_with_markers(sections, markers=share.INPUT_MARKERS)

            # process new output
            new_output_text = utils.get_code_from_diff(raw_editable_section, output_diff)
            new_output_text = utils.add_marker_around_output(new_output_text, markers=share.EDITABLE_MARKERS)

            input_output_diff = get_unidiff(new_editable_section, new_output_text, fromfile='input', tofile='output')

            data = {
                'input': new_input_text,
                'output': new_output_text,
                'input_output_diff': input_output_diff,
                'event': event_text,
                'raw_editable_section': raw_editable_section
            }
            result.append(data)

    return result

def main():
    raw_data = utils.read_json(RAW_FILE)
    new_data = []
    random.seed(42)
    for obj in tqdm(raw_data, total=len(raw_data)):
        new_train_samples = process(obj['input'], obj['output'], no_ref=True)
        for sample in new_train_samples:
            new_obj = copy.deepcopy(obj)

            new_obj['instruction'] = share.INSTRUCTION
            new_obj['input'] = sample['event'] + '\n\n' + sample['input']
            new_obj['output'] = sample['output']
            new_obj['input_output_diff'] = sample['input_output_diff']
            new_obj['event'] = sample['event']
            new_obj['raw_editable_section'] = sample['raw_editable_section']
            new_obj['raw_output'] = obj['output']

            new_data.append(new_obj)
    random.shuffle(new_data)
    utils.write_json(new_data, NEW_FILE)

if __name__ == '__main__':
    main()
