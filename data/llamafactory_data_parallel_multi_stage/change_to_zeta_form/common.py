# utils to encode and decode code changes into CodeT5 format.

import copy
import ast
from abc import ABC, abstractmethod
from textwrap import dedent
from pathlib import Path
from typing import cast, Iterable, TypeVar, Callable, Generic, Any
from dataclasses import dataclass
from transformers import AutoTokenizer

# 定义缓存目录
cache_dir = "model/"

# 定义断言函数
def assert_eq(a, b):
    assert a == b, f"{a} != {b}"

# 定义关闭tokenizer警告的函数
def _turn_off_tokenizer_warning(tokenizer):
    """关闭tokenizer的警告信息"""
    pass

TokenizerType = AutoTokenizer
Token = int
TokenSeq = list[Token]
T1 = TypeVar("T1")
T2 = TypeVar("T2")
E1 = TypeVar("E1", covariant=True)

try:
    # 使用本地已下载的qwen2.5-coder-3b模型
    local_model_path = "/data/mnt_bucket/qzq/CustomTrain/output/gsn/custom_train_new_stage_data/qzq_run_sft_0.5b_order_stage/checkpoint-17000"
    
    if Path(local_model_path).exists():
        print(f"✅ 使用本地qwen2.5-coder-3b模型: {local_model_path}", flush=True)
        _BaseTokenizer = cast(
            TokenizerType, TokenizerType.from_pretrained(
                local_model_path,
                local_files_only=True,  # 只使用本地文件
                trust_remote_code=True
            )
        )
        print("✅ 成功加载本地 qwen2.5-coder-3b tokenizer", flush=True)
    else:
        raise FileNotFoundError(f"本地模型路径不存在: {local_model_path}")
        
except Exception as e:
    print(f"⚠️  无法加载本地qwen2.5-coder-3b tokenizer: {e}")


_turn_off_tokenizer_warning(_BaseTokenizer)

Add = "<add>"
Del = "<del>"
BOS = "<s>"
EOS = "</s>"
PAD = "<pad>"
N_Extra_Ids = 100
add_token_list = [Add, Del, BOS, EOS, PAD]
extra_token_list = []

for i in range(N_Extra_Ids):
    add_token_list.append(f"<extra_id_{i}>")
    extra_token_list.append(f"<extra_id_{i}>")

"""
`_BaseTokenizer` extended with <add> and <del> tokens.
Note that you should avoid using _Tokenizer.encode(text) directly as it
will incorrectly eat the spaces around <add> and <del>.
Use `encode_change` instead.
"""
_Tokenizer = copy.deepcopy(_BaseTokenizer)
_Tokenizer.add_tokens(add_token_list)


def get_tk_id(token: str) -> int:
    "Convert a token str into the corresponding integer index."
    seq = _Tokenizer.encode(token, add_special_tokens=False)
    assert len(seq) == 1
    id = seq[0]
    assert_eq(_Tokenizer.decode([id], add_special_tokens=False), token)
    return id


Add_id = get_tk_id(Add)
Del_id = get_tk_id(Del)
# 获取换行符的token ID
newline_tokens = _Tokenizer.encode("\n", add_special_tokens=False)
if len(newline_tokens) == 1:
    Newline_id = newline_tokens[0]
else:
    print(f"警告: 换行符被编码为多个token: {newline_tokens}")
    Newline_id = newline_tokens[0] if newline_tokens else -1
BOS_id = get_tk_id("<s>")
EOS_id = get_tk_id("</s>")
PAD_id = get_tk_id("<pad>")
Extra_id_list = [get_tk_id(token) for token in extra_token_list]


def is_extra_id(tk: int) -> bool:
    return tk in Extra_id_list

def split_list(
    lst: list[T1],
    sep: T1,
) -> list[list[T1]]:
    """
    Split a list into segments by a separator, always ends with an empty list.
    """
    if not lst:
        return []
    result = list[list[T1]]()
    ptr = 0
    for i, item in enumerate(lst):
        if item == sep:
            result.append(lst[ptr:i])
            ptr = i + 1
    result.append(lst[ptr:])
    return result

def join_list(
    segs: Iterable[Iterable[T1]],
    sep: T1 | None = None,
) -> list[T1]:
    result = list[T1]()
    for i, seg in enumerate(segs):
        if sep is not None and i > 0:
            result.append(sep)
        result.extend(seg)
    return result

def tk_splitlines(tks: TokenSeq):
    """
    按行分割token序列
    """
    if not tks:
        return []

    try:
        # 解码整个token序列
        full_text = _Tokenizer.decode(tks, add_special_tokens=False, clean_up_tokenization_spaces=False)
        
        # 按行分割
        text_lines = full_text.split('\n')
        
        # 重新编码每一行
        token_lines = []
        for line in text_lines:
            if line:  # 非空行
                line_tokens = _Tokenizer.encode(line, add_special_tokens=False)
                token_lines.append(line_tokens)
            else:  # 空行
                token_lines.append([])
        
        return token_lines
        
    except Exception as e:
        print(f"解码/编码过程中出错: {e}")
        # 如果出错，返回原始序列作为单行
        return [tks]

def output_ids_as_seqs(output_ids: Iterable[Token]) -> dict[Token, TokenSeq]:
    """Parse the CodeT5 model's output as a series of key-value pairs.
    <pad>, <mask>, or <s> or </s> tokens are filtered out."""
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

def inline_output_tokens(
    input: TokenSeq, output: TokenSeq, leave_unpredicted=False
) -> TokenSeq:
    """Inline CodeT5's output tokens into its input tokens."""
    out_map = output_ids_as_seqs(output)
    combined = TokenSeq()
    for tk in input:
        if is_extra_id(tk):
            if tk in out_map:
                combined.extend(out_map[tk])
            elif leave_unpredicted:
                combined.append(tk)
        else:
            combined.append(tk)
    return combined

def decode_tokens(tokens: TokenSeq, prettify: bool = False) -> str:
    text = _Tokenizer.decode(
        tokens, add_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if prettify:
        text = text.replace("<extra_id_", "<mask_")
    return text

class _ChangeBase(Generic[E1]):

    @property
    @abstractmethod
    def earlier(self) -> E1:
        ...

    @property
    @abstractmethod
    def later(self) -> E1:
        ...

    @property
    def changed(self) -> bool:
        return True

@dataclass(frozen=True)
class Modified(_ChangeBase[E1]):
    before: E1
    after: E1
    # Used for optimization. If False, `before`` may still equal to `after`.
    unchanged: bool = False

    def map(self, f: Callable[[E1], T2]) -> "Modified[T2]":
        if self.unchanged:
            return Modified.from_unchanged(f(self.before))
        else:
            return Modified(f(self.before), f(self.after))

    def inverse(self) -> "Modified[E1]":
        return Modified(self.after, self.before)

    @property
    def earlier(self) -> E1:
        return self.before

    @property
    def later(self) -> E1:
        return self.after

    @property
    def changed(self) -> bool:
        return not self.unchanged

    @staticmethod
    def as_char():
        return "M"

    @staticmethod
    def from_unchanged(v: T1) -> "Modified[T1]":
        return Modified(v, v, unchanged=True)

    def __repr__(self):
        if self.before == self.after:
            return f"Modified(before=after={repr(self.before)})"
        else:
            return f"Modified(before={repr(self.before)}, after={repr(self.after)})"

def tokens_to_change(tokens: TokenSeq) -> Modified[str]:
    """
    将token序列转换为代码变化
    使用字符串处理方式，按行判断<add>和<del>标记
    """
    # 将tokens解码为字符串
    full_text = decode_tokens(tokens)
    
    # 按行分割
    lines = full_text.split('\n')
    
    before_lines = []
    after_lines = []
    
    for line in lines:
        if '<add>' in line:
            # 包含<add>标记的行，添加到after_code中
            # 移除<add>标记，保留后面的内容
            # 计算前导空格数
            after_content = line.lstrip().replace('<add>', '')
            if after_content:
                # 计算前导空格数
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
    
    return Modified(before_code, after_code)

def as_any(x) -> Any:
    return x

def normalize_code_by_ast(
    code: str, sort_keyargs: bool = True, remove_doc_string: bool = True
) -> str:
    """Normalize the code by parsing and unparsing it using the AST module.
    If parsing fails, return the original code."""

    class KeyargSorter(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            if node.keywords:
                node.keywords.sort(key=lambda x: x.arg or "None")
            return node

    class DocStringremover(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            return self._visit_def(node)

        def visit_Module(self, node: ast.Module) -> Any:
            return self._visit_def(node)

        def visit_ClassDef(self, node: ast.ClassDef):
            return self._visit_def(node)

        def _visit_def(self, node):
            node = as_any(self.generic_visit(node))
            match node.body:
                case [ast.Expr(value=ast.Constant(value=str())), *body]:
                    node.body = body
            return node

    try:
        tree = ast.parse(dedent(code))
        if remove_doc_string:
            tree = DocStringremover().visit(tree)
        if sort_keyargs:
            tree = KeyargSorter().visit(tree)
        return ast.unparse(tree)
    except (SyntaxError, ValueError):
        return code
