import json
import re

from tokenizer import _Tokenizer, inline_output_tokens, tokens_to_change


"""
literal string processing
"""
def str_to_token_num(s: str | None) -> int:
    if s is None:
        return 0
    return len(_Tokenizer.encode(s, add_special_tokens=False))

def cutoff_str_by_token_num(s: str, max_len: int) -> str:
    seq = _Tokenizer.encode(s, add_special_tokens=False)[:max_len]
    output = _Tokenizer.decode(seq, add_special_tokens=False)
    return output

"""
json and jsonl processing functions
"""
def read_json(json_file: str) -> dict | list:
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(data: dict | list, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f: # UTF-8 for chinese characters
        json.dump(data, f, indent=2, ensure_ascii=False) # for non-ASCII characters

def read_jsonl(jsonl_file: str) -> list:
    data = []
    with open(jsonl_file, encoding='utf-8') as f:
        for _, line in enumerate(f.readlines()):
            data.append(json.loads(line))
    return data

"""
'input' field in dataset processing functions
"""
def extract_sections_from_input(input_text: str,
                                markers: dict[str, str] = {
                                    'start': '### User Excerpt:\n\n',
                                    'reference': '### User Edits Reference:\n\n',
                                    'response': '### Response:\n\n'
                                },
                                return_editable_section: bool = True) -> dict[str, str]:
    """Extract different code sections from 'input' field in dataset.

    Args:
        input_text (str): 'input' field in dataset
        markers (dict): template markers for splitting input_text
        return_editable_section (bool):
            If True, extract the editable section from main section and return it. If False, return None.

    Returns:
        A dictionary which has 'main_section', 'reference_section', 'editable_section' keys.
        If there is no corresponding section, the value is None.
    """
    start_marker = markers['start']
    reference_marker = markers['reference']
    response_marker = markers['response']

    start_pos = input_text.find(start_marker)
    reference_pos = input_text.find(reference_marker)
    response_pos = input_text.find(response_marker)

    if start_pos == -1:
        raise RuntimeError('No start marker found')
    if response_pos == -1:
        raise RuntimeError('No response marker found')

    # main section
    if reference_pos != -1:
        main_section = input_text[start_pos + len(start_marker):reference_pos]
    else:
        main_section = input_text[start_pos + len(start_marker):response_pos]

    # reference section
    reference_section = ''
    if reference_pos != -1:
        reference_section = input_text[reference_pos + len(reference_marker):response_pos]

    # extra id section
    def extract_extra_id_section(main_section: str) -> str:
        extra_id_pattern = r'<extra_id_\d+>'

        main_section_lines = main_section.split("\n")

        # Track positions of extra_id lines
        extra_id_indices = []
        for i, line in enumerate(main_section_lines):
            if re.search(extra_id_pattern, line):
                extra_id_indices.append(i)

        if not extra_id_indices:
            return ''

        start_index = extra_id_indices[0]
        end_index = None

        for i in range(extra_id_indices[-1] + 1, len(main_section_lines)):
            if '<add>' not in main_section_lines[i] and '<del>' not in main_section_lines[i]:
                end_index = i
                break

        # If all remaining lines have <add> or <del>
        if end_index is None:
            end_index = len(main_section_lines)

        section = '\n'.join(main_section_lines[start_index:end_index])
        return section

    extra_id_section = ''
    if return_editable_section:
        extra_id_section = extract_extra_id_section(main_section)

    return {
        'main_section': main_section,
        'reference_section': reference_section,
        'editable_section': extra_id_section
    }

def add_marker_around_editable_section(editable_section: str,
                                       markers: dict[str, str] = {
                                           'start': '<extra_id_start>\n',
                                           'end': '\n<extra_id_end>'
                                        }) -> str:
    """Remove all `<extra_id_>` tags, and add start and end marker around editable section."""
    if not editable_section:
        return ''

    extra_id_pattern = r'<extra_id_\d+>'
    start_marker = markers['start']
    end_marker = markers['end']

    # Remove all <extra_id_\d+> tags
    new_extra_id_section = re.sub(extra_id_pattern, '', editable_section)

    # Add markers
    new_extra_id_section = start_marker + new_extra_id_section + end_marker
    return new_extra_id_section

def replace_editable_section(main_section: str, old_section: str, new_section: str) -> str:
    """Replace the old editable section in main section with new one and return the new main section."""
    return main_section.replace(old_section, new_section)

def concat_sections_with_markers(sections: dict[str, str],
                                 markers: dict[str, str] = {
                                    'start': '### User Excerpt:\n\n',
                                    'reference': '### User Edits Reference:\n\n',
                                    'response': '### Response:\n\n'
                                }) -> str:
    """Concatenate main section and reference section with specific markers.

    Args:
        sections (dict[str, str]): a dictionary which has 'main_section' and 'reference_section' keys
        markers (dict[str, str]): a dictionary which has 'start', 'reference' and 'response' keys

    Returns:
        a concatenated string representing the new 'input' field in dataset
    """
    start_marker = markers['start']
    reference_marker = markers['reference']
    response_marker = markers['response']

    if 'main_section' not in sections or not sections['main_section']:
        raise RuntimeError('No main_section key found or the value is None')

    output = start_marker + sections["main_section"] + '\n\n'

    if 'reference_section' in sections and sections['reference_section']:
        output += reference_marker + sections["reference_section"] + '\n\n'

    output += response_marker

    return output

"""
'output' field (ground truth) in dataset processing functions
"""
def get_code_from_diff(editable_section: str, diff: str) -> str:
    """Apply code diff to the editable section. The editable section and code diff must be in `<extra_id_>` style.

    Args:
        tokenizer: AutoTokenizer from transformers lib
        editable_section (str): code section
        diff (str): code diff applying to the editable section

    Returns:
        code snippet with code diff applied
    """
    # 使用tokenizer将文本转换为token_ids
    editable_section_tokens = _Tokenizer.encode(editable_section, add_special_tokens=False)
    diff_tokens = _Tokenizer.encode(diff, add_special_tokens=False)

    _, after_code = tokens_to_change(
        inline_output_tokens(editable_section_tokens, diff_tokens)
    )
    return after_code

def add_marker_around_output(output_section: str,
                             markers: dict[str, str] = {
                                 'start': '<extra_id_start>\n',
                                 'end': '\n<extra_id_end>'
                                 }) -> str:
    """Add start and end marker around output section."""
    start_marker = markers['start']
    end_marker = markers['end']

    # Add markers
    new_output_section = start_marker + output_section + end_marker
    return new_output_section
