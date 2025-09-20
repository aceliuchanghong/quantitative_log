# -*- coding: utf-8 -*-
import json
import re
from typing import Any, Callable, List, Dict, Optional


def _replace_new_line(match: re.Match[str]) -> str:
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r"\"", value)

    return match.group(1) + value + match.group(3)


def _custom_parser(multiline_string: str) -> str:
    """
    The LLM response for `action_input` may be a multiline
    string containing unescaped newlines, tabs or quotes. This function
    replaces those characters with their escaped counterparts.
    (newlines in JSON must be double-escaped: `\\n`)
    """
    if isinstance(multiline_string, (bytes, bytearray)):
        multiline_string = multiline_string.decode()

    # Convert single-quoted keys to double-quoted keys
    multiline_string = re.sub(r"'(\w+)'", r'"\1"', multiline_string)

    # Convert single-quoted values to double-quoted values
    multiline_string = re.sub(r":\s*'([^']*)'", r': "\1"', multiline_string)

    multiline_string = re.sub(
        r'("action_input"\:\s*")(.*?)(")',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )

    return multiline_string


def parse_partial_json(s: str, *, strict: bool = False) -> Any:
    """Parse a JSON string that may be missing closing braces.

    Args:
        s: The JSON string to parse.
        strict: Whether to use strict parsing. Defaults to False.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    # Attempt to parse the string as-is.
    try:
        return json.loads(s, strict=strict)
    except json.JSONDecodeError:
        pass

    # Initialize variables.
    new_chars = []
    stack = []
    is_inside_string = False
    escaped = False

    # Process each character in the string one at a time.
    for char in s:
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == "\n" and not escaped:
                char = "\\n"  # Replace the newline character with the escape sequence.
            elif char == "\\":
                escaped = not escaped
            else:
                escaped = False
        else:
            if char == '"':
                is_inside_string = True
                escaped = False
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char == "}" or char == "]":
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    # Mismatched closing character; the input is malformed.
                    return None

        # Append the processed character to the new string.
        new_chars.append(char)

    # If we're still inside a string at the end of processing,
    # we need to close the string.
    if is_inside_string:
        new_chars.append('"')

    # Reverse the stack to get the closing characters.
    stack.reverse()

    # Try to parse mods of string until we succeed or run out of characters.
    while new_chars:
        # Close any remaining open structures in the reverse
        # order that they were opened.
        # Attempt to parse the modified string as JSON.
        try:
            return json.loads("".join(new_chars + stack), strict=strict)
        except json.JSONDecodeError:
            # If we still can't parse the string as JSON,
            # try removing the last character
            new_chars.pop()

    # If we got here, we ran out of characters to remove
    # and still couldn't parse the string as JSON, so return the parse error
    # for the original string.
    return json.loads(s, strict=strict)


_json_markdown_re = re.compile(r"```(json)?(.*)", re.DOTALL)


def parse_json_markdown(
    json_string: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> dict:
    """Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    try:
        return _parse_json(json_string, parser=parser)
    except json.JSONDecodeError:
        # Try to find JSON string within triple backticks
        match = _json_markdown_re.search(json_string)

        # If no match found, assume the entire string is a JSON string
        if match is None:
            json_str = json_string
        else:
            # If match found, use the content within the backticks
            json_str = match.group(2)
    return _parse_json(json_str, parser=parser)


_json_strip_chars = " \n\r\t`"


def _parse_json(
    json_str: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> dict:
    # Strip whitespace,newlines,backtick from the start and end
    json_str = json_str.strip(_json_strip_chars)

    # handle newlines and other special characters inside the returned value
    json_str = _custom_parser(json_str)

    # Parse the JSON string into a Python dictionary
    return parser(json_str)


def parse_and_check_json_markdown(
    text: str, expected_keys: List[str] = ["is_real_time", "is_nsfw"]
) -> dict:
    """
    Parse a JSON string from a Markdown string and check that it
    contains the expected keys.

    Args:
        text: The Markdown string.
        expected_keys: The expected keys in the JSON string.

    Returns:
        The parsed JSON object as a Python dictionary.

    Raises:
        ValueError: If the JSON string is invalid or does not contain
            the expected keys.
    """
    try:
        json_obj = parse_json_markdown(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Got invalid JSON object. Error: {e}")
    for key in expected_keys:
        if key not in json_obj:
            raise ValueError(
                f"Got invalid return object. Expected key `{key}` "
                f"to be present, but got {json_obj}"
            )
    return json_obj


def parse_and_check_json_list(
    text: str, expected_keys: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """
    从文本中提取 JSON 列表，支持去除尾随逗号，并验证每个 dict 包含 expected_keys。
    """
    # 匹配最外层的数组，从第一个 '[' 开始，到最后一个 ']' 结束
    bracket_stack = []
    start = -1
    for i, char in enumerate(text):
        if char == "[":
            if start == -1:
                start = i
            bracket_stack.append(char)
        elif char == "]":
            if bracket_stack:
                bracket_stack.pop()
                if not bracket_stack:  # 完全匹配
                    json_str = text[start : i + 1]
                    break
    else:
        print("Error: No balanced JSON list found in the text.")
        return None

    # 预处理：去除 JSON 中的尾随逗号（在 }, 后面或 ] 前的 ,）
    # 先去除空格/换行，再匹配
    def remove_trailing_commas(s: str) -> str:
        # 去除在 } 后面、] 前面的逗号，后面是空或空白+]
        s = re.sub(r",\s*(?=\])", "", s)  # ,[空格]]
        s = re.sub(r",\s*(?=\})", "", s)  # ,}
        return s

    cleaned_json_str = remove_trailing_commas(json_str)

    try:
        data = json.loads(cleaned_json_str)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON after cleaning. {e}")
        return None

    if not isinstance(data, list):
        print("Error: Extracted JSON is not a list.")
        return None

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Error: Item at index {idx} is not a dictionary.")
            return None
        missing_keys = [key for key in expected_keys if key not in item]
        if missing_keys:
            print(f"Error: Item at index {idx} is missing keys: {missing_keys}")
            return None

    return data


if __name__ == "__main__":
    # uv run z_utils/get_json.py
    text = """```json\n{"name":"张三", "age": 27, "爱好": ["羽毛球"""
    text2 = """
    {'translate_result': 'iPad 呢？'}
    """
    text3 = """
        ```json{
      "question": "合同里面SOB或者SOA编号是？格式是SOB20...",
      "answer": "SOB202102-14875"
    }``` 以上就是结果
        """
    text4 = """
    ca`sc
    [
        {"question": "我不太明白，为什么皇帝自己不处理那些奏折，全都丢给他呢？", "answer": "书里提到，皇帝似乎更专注于宏大的天道运行，将具体的日常政务交由宰相来处理，这是一种权力分工。"},
        {"question": "那些神仙听起来好像都在摸鱼，他们都不怕被惩罚吗？", "answer": "从文本来看，天庭似乎缺少严格的监督和惩罚体系，导致大家普遍工作不积极，所以他们看起来并不担心。"}
    ]
    f''c``s
    """
    xx = parse_json_markdown(text)
    print(xx)
    yy = parse_json_markdown(text2)
    print(f"{yy}")
    zz = parse_and_check_json_markdown(text3, ["question", "answer"])
    print(zz)
    aa = parse_and_check_json_list(text4, ["question", "answer"])
    print(aa)
