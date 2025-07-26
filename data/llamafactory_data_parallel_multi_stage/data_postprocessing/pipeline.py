import copy
from tqdm import tqdm
from utils import *

from share import EDITABLE_MARKERS, INPUT_MARKERS, NEW_DATA, RAW_DATA

def main():
    raw_data = read_json(RAW_DATA)

    new_data = []
    for obj in tqdm(raw_data, total=len(raw_data)):
        new_obj = copy.deepcopy(obj)

        sections = extract_sections_from_input(new_obj['input'], INPUT_MARKERS)

        # process main section
        new_editable_section = add_marker_around_editable_section(sections['editable_section'], EDITABLE_MARKERS)
        sections['main_section'] = replace_editable_section(sections['main_section'], sections['editable_section'], new_editable_section)
        new_obj['input'] = concat_sections_with_markers(sections, INPUT_MARKERS)

        # process output
        code = get_code_from_diff(sections['editable_section'], new_obj['output'])
        new_output_section = add_marker_around_output(code, EDITABLE_MARKERS)
        new_obj['output'] = new_output_section

        new_data.append(new_obj)

    write_json(new_data, NEW_DATA)

if __name__ == '__main__':
    main()
