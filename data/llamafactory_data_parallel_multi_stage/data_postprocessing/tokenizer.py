import copy
import sys
from pathlib import Path
from typing import Iterable

from transformers import AddedToken, AutoTokenizer


# 获取当前文件的目录和父目录
current_dir = Path(__file__).parent  # 当前文件所在目录
parent_dir = current_dir.parent      # 当前文件的父目录
sys.path.append(str(parent_dir))

from share import MODEL_PATH


"""
Special tokens initialization
"""
Token = int
TokenSeq = list[Token]

Add = "<add>"
Del = "<del>"
BOS = "<s>"
EOS = "</s>"
PAD = "<pad>"
N_Extra_Ids = 100
add_token_list: list[str | AddedToken] = [Add, Del, BOS, EOS, PAD]
extra_token_list = []

for i in range(N_Extra_Ids):
    add_token_list.append(f"<extra_id_{i}>")
    extra_token_list.append(f"<extra_id_{i}>")

"""
Initialize _BaseTokenizer from TOKENIZER_PATH
"""
try:
    local_path = MODEL_PATH
    if Path(local_path).exists():
        print(f"✅ 使用本地 qwen2.5-coder 模型: {local_path}")
        _BaseTokenizer = AutoTokenizer.from_pretrained(
            local_path,
            local_files_only=True, # 只使用本地文件
            trust_remote_code=True
        )
        print("✅ 成功加载本地 qwen2.5-coder tokenizer")
    else:
        raise FileNotFoundError(f"本地模型路径不存在: {local_path}")

except Exception as e:
    raise RuntimeError(f"⚠️ 无法加载本地 qwen2.5-coder tokenizer: {e}")

"""
`_BaseTokenizer` extended with <add> and <del> tokens.
Note that you should avoid using _Tokenizer.encode(text) directly as it
will incorrectly eat the spaces around <add> and <del>.
Use `encode_change` instead.
"""
_Tokenizer = copy.deepcopy(_BaseTokenizer)
_Tokenizer.add_tokens(add_token_list)

"""
Convert special tokens to token id
"""
def get_tk_id(token: str) -> int:
    """Convert a token str into the corresponding integer index."""
    seq = _Tokenizer.encode(token, add_special_tokens=False)
    assert len(seq) == 1
    id = seq[0]
    assert _Tokenizer.decode([id], add_special_tokens=False) == token
    return id

Add_id = get_tk_id(Add)
Del_id = get_tk_id(Del)
BOS_id = get_tk_id(BOS)
EOS_id = get_tk_id(EOS)
PAD_id = get_tk_id(PAD)

# 获取换行符的token ID
newline_tokens = _Tokenizer.encode("\n", add_special_tokens=False)
if len(newline_tokens) == 1:
    Newline_id = newline_tokens[0]
else:
    print(f"警告: 换行符被编码为多个token: {newline_tokens}")
    Newline_id = newline_tokens[0] if newline_tokens else -1

Extra_id_list = [get_tk_id(token) for token in extra_token_list]

def is_extra_id(tk: int) -> bool:
    return tk in Extra_id_list

"""
Utilization
"""
def output_ids_as_seqs(output_ids: Iterable[Token]) -> dict[Token, TokenSeq]:
    """Parse the CodeT5 model's output as a series of key-value pairs. <pad>, <mask>, or <s> or </s> tokens are filtered out."""
    buff = TokenSeq()
    key = None
    seqs = dict[Token, TokenSeq]()

    for tk in output_ids:
        if tk <= 0 or tk == BOS_id or tk == EOS_id:
            continue  # pad, mask token, or sequence token
        if tk in Extra_id_list:
            if key is not None:
                seqs[key] = buff
            buff = TokenSeq()
            key = tk
        else:
            buff.append(tk)
    if key is not None:
        seqs[key] = buff
    return seqs

def inline_output_tokens(input: TokenSeq, output: TokenSeq, leave_unpredicted=False) -> TokenSeq:
    """Inline CodeT5's output tokens into its input tokens."""
    out_map = output_ids_as_seqs(output)
    combined = []
    for tk in input:
        if is_extra_id(tk):
            if tk in out_map:
                combined.extend(out_map[tk])
            elif leave_unpredicted:
                combined.append(tk)
        else:
            combined.append(tk)
    return combined

def tokens_to_change(tokens: TokenSeq) -> tuple[str, str]:
    """将token序列转换为代码变化, 使用字符串处理方式, 按行判断<add>和<del>标记."""
    # 将tokens解码为字符串
    full_text = _Tokenizer.decode(tokens, add_special_tokens=False, clean_up_tokenization_spaces=False)

    # 按行分割
    lines = full_text.split('\n')

    before_lines = []
    after_lines = []

    for line in lines:
        if '<add>' in line:
            # 包含<add>标记的行，添加到after_code中
            # 移除<add>标记，保留后面的内容
            after_content = line.lstrip().replace('<add>', '')
            if after_content:
                after_lines.append(after_content)
        elif '<del>' in line:
            # 包含<del>标记的行，添加到before_code中
            # 移除<del>标记，保留后面的内容
            before_content = line.lstrip().replace('<del>', '')
            if before_content:
                before_lines.append(before_content)
        else:
            # 普通行，同时添加到before和after中
            if line:
                before_lines.append(line)
                after_lines.append(line)

    # 重新组合为代码字符串
    before_code = '\n'.join(before_lines)
    after_code = '\n'.join(after_lines)

    return (before_code, after_code)
