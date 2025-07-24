import copy

from tqdm import tqdm
from utils import *


RAW_DATA_DIRECTORY = ""
OUTPUT_DATA_DIRECOTRY = ""

DATA_FILE = "debug.json"

def main():
    raw_data = read_json(f"{RAW_DATA_DIRECTORY}/{DATA_FILE}")
    output_file = f"{OUTPUT_DATA_DIRECOTRY}/{DATA_FILE}"

    new_data = []
    for obj in tqdm(raw_data, total=len(raw_data)):
        new_obj = copy.deepcopy(obj)

        sections = extract_sections_from_input(new_obj['input'])

        # process main section
        new_editable_section = add_marker_around_editable_section(sections['editable_section'])
        sections['main_section'] = replace_editable_section(sections['main_section'], sections['editable_section'], new_editable_section)
        new_obj['input'] = concat_sections_with_markers(sections)

        # process output
        code = get_code_from_diff(sections['editable_section'], new_obj['output'])
        new_output_section = add_marker_around_output(code)
        new_obj['output'] = new_output_section

        new_data.append(new_obj)

    write_json(new_data, output_file)

if __name__ == '__main__':
    main()
