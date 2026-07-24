import torch
import os 
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageFont, ImageDraw
import json 
import re
import math 
import cv2 
import argparse
import time 
from tqdm import tqdm  # 用于显示进度条


def inference(img_url, prompt="QwenVL HTML"):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_url,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt")
    inputs = inputs.to(model.device)
    
    # 模型推理
    generated_ids = model.generate(**inputs, max_new_tokens=8192, temperature=0.0)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # 输出为html格式
    html_output = output_text[0]
    return html_output


def remove_lines_starting_with(text):
    lines = text.splitlines(keepends=True)
    
    filtered = []
    prefixes_to_remove = ('Z:')
    
    for line in lines:
        stripped = line.lstrip()
        if not stripped.strip():
            continue
        if stripped.startswith(prefixes_to_remove):
            continue

        filtered.append(line)  
    return "".join(filtered)

def process_code_content(content: str) -> str:
    content = content.replace('```', '')
    
    content = re.sub(r'^\s*<pre[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</pre>\s*$', '', content, flags=re.IGNORECASE)
    
    content = re.sub(r'^\s*<code[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</code>\s*$', '', content, flags=re.IGNORECASE)
    
    # 包裹在 ``` 中
    return f"```code\n{content.strip()}\n```"


def process_pseudocode_content(content: str) -> str:
    """Process pseudocode content, preserving indentation and not breaking LaTeX formulas"""

    content = content.replace('```', '')
    content = re.sub(r'^\s*<(pre|code)[^>]*>', '', content, flags=re.IGNORECASE | re.MULTILINE)
    content = re.sub(r'</(pre|code)>\s*$', '', content, flags=re.IGNORECASE | re.MULTILINE)

    # Extract and protect LaTeX formulas
    math_blocks = []
    def save_math(match):
        placeholder = f"___MATH_ID_{len(math_blocks)}___"
        math_blocks.append(match.group(0))
        return placeholder

    # Regex: prioritize matching double dollar signs, then single dollar signs
    math_pattern = r'(\$\$.*?\$\$|\$.*?\$)'
    protected_content = re.sub(math_pattern, save_math, content, flags=re.DOTALL)

    protected_content = protected_content.replace(' ', '&nbsp;')
    protected_content = protected_content.replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
    protected_content = protected_content.replace('\n', '<br>')

    # Restore LaTeX formulas
    final_content = protected_content
    for i, original_math in enumerate(math_blocks):
        placeholder = f"___MATH_ID_{i}___"
        final_content = final_content.replace(placeholder, original_math)

    return f"___\n<br>{final_content.strip()}<br>\n___"


def qwenvl_cast_html_tag(input_text: str) -> str:
    output = input_text
    IMG_RE = re.compile(
        r'<img\b[^>]*\bdata-bbox\s*=\s*"?\d+,\d+,\d+,\d+"?[^>]*\/?>',
        flags=re.IGNORECASE,
    )
    output = IMG_RE.sub('', output)

    # code
    def replace_code(match):
        content = match.group(1)
        processed_content = process_code_content(content)
        return f"\n\n{processed_content}\n\n"
    
    code_pattern = re.compile(
        r'<div\b[^>]*class="code"[^>]*>(.*?)</div>',
        flags=re.DOTALL | re.IGNORECASE,
    )
    output = code_pattern.sub(replace_code, output)
    
    # pseudocode
    def replace_pseudocode(match):
        content = match.group(1)
        processed_content = process_pseudocode_content(content)
        return f"\n\n{processed_content}\n\n"
    
    pseudocode_pattern = re.compile(
        r'<div\b[^>]*class="pseudocode"[^>]*>(.*?)</div>',
        flags=re.DOTALL | re.IGNORECASE,
    )
    output = pseudocode_pattern.sub(replace_pseudocode, output)

    # <div>
    def strip_div(class_name: str, txt: str) -> str:
        if class_name in ['code', 'pseudocode']:
            return txt

        def replace_func(match):
            content = match.group(1)
            
            # 仅针对匹配到的 div 内部内容进行清洗
            if class_name == 'chart':
                content = re.sub(r'^\s*(click\s+|style\s+|linkStyle\s+|stroke|classDef\s+|class\s+)\b.*\n?', '', content, flags=re.MULTILINE | re.IGNORECASE)
                content = re.sub(r'^\s*(?:%%|::icon).*\n?', '', content, flags=re.MULTILINE)

                content = content.strip()
                if content.startswith('mermaid'):
                    content = '```' + content
                elif re.match(r'^```\s*mermaid', content):
                    pass
                else:
                    content = '```mermaid\n' + content
                if not content.endswith('```'):
                    content += '\n```'

            if class_name == 'music':
                content = remove_lines_starting_with(content)

                content = content.strip()
                if content.startswith('abc'):
                    content = '```' + content
                elif re.match(r'^```\s*abc', content):
                    pass
                else:
                    content = '```abc\n' + content
                if not content.endswith('```'):
                    content += '\n```'
                
            return f"\n\n{content}\n\n"

        pattern = re.compile(
            rf'\s*<div\b[^>]*class="{class_name}"[^>]*>(.*?)</div>\s*',
            flags=re.DOTALL | re.IGNORECASE,
        )
        return pattern.sub(replace_func, txt)

    other_classes = ['image', 'chemistry', 'table', 'formula', 'image caption', 'table caption']
    for cls in other_classes:
        output = strip_div(cls, output)
    
    # <p>
    output = re.sub(
        r'<p\b[^>]*>(.*?)</p>',
        r'\n\n\1\n\n',
        output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    
    output = output.replace(" </td>", "</td>")
    return output


def smart_resize(
    height: int, width: int, factor: int = 32, min_pixels: int = 3136, max_pixels: int = 7200*32*32
):
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def plot_bbox(img_path, pred, output_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    img_height, img_width, _ = img.shape
    bboxes = []
    pattern = re.compile(r'data-bbox="(\d+),(\d+),(\d+),(\d+)"')

    def replace_bbox(match):
        x1, y1, x2, y2 = map(int, match)
        x1, y1 = int(x1/1000 * img_width), int(y1/1000 * img_height)
        x2, y2 = int(x2/1000 * img_width), int(y2/1000 * img_height)
        bboxes.append([x1,y1,x2,y2])

    matches = re.findall(pattern, pred)
    if matches:
        for match in matches:
            replace_bbox(match)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 8)
    cv2.imwrite(output_path, img)


