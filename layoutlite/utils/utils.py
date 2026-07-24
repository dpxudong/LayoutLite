import datetime
import logging
import logging.handlers
import os
import sys
import torch
import json, cv2
import numpy as np
from transformers import StoppingCriteria
import matplotlib.pyplot as plt


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)



def find_repeating_pattern_fast(text, min_len=2, max_len=20, min_repeat=15):
    """
    只检测文本尾部的连续重复子串
    :return: (pattern, repeat_count) or None
    """
    n = len(text)

    for l in range(min_len, max_len + 1):
        # 至少要满足最小重复长度
        if n < l * min_repeat:
            continue

        # 从尾部开始取一个pattern
        sub = text[n - l:n]

        count = 1
        pos = n - l

        # 向前检查
        while pos - l >= 0 and text[pos - l:pos] == sub:
            count += 1
            pos -= l
            if count >= min_repeat:
                return sub, count

    return None




def analyze_single_text_fast(text, min_len=2, max_len=50, min_repeat=20):
    """
    快速分析单个 OCR 输出字符串，发现第一个异常即返回
    :param text: OCR 识别结果字符串
    :return: 是否发现异常
    """
    result = find_repeating_pattern_fast(text, min_len, max_len, min_repeat)
    if result:
        pattern, count = result
        # print(f"❗ Detected repeating pattern: '{pattern}' x {count}")
        return True
    else:
        # print("✅ No significant repeating pattern found.")
        return False



class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if len(outputs) > 300:
                    is_repeat = analyze_single_text_fast(outputs)
                    if is_repeat:
                        return True
                if len(outputs) > 4096*3:
                    return True
            
                if keyword in outputs:
                    return True
        return False



def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # # num_new_tokens = 1
    # # tokenizer.add_tokens(special_tokens_dict, special_tokens=True)
    # model.resize_token_embeddings(len(tokenizer))

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    

def save_score_heatmap(scores, image_path, output_image_dir, step):
    """
    scores: 1D torch.Tensor
    image_path: str
    output_image_dir: str
    """

    scores = (scores - scores.mean()) / scores.std()

    # 映射到 0~1
    scores = torch.sigmoid(scores)

    os.makedirs(output_image_dir, exist_ok=True)

    # 读取图片
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # grid 总数
    n = scores.numel()

    # 推测 grid 行列数
    aspect = h / w
    grid_h = int(round(np.sqrt(n * aspect)))
    grid_w = int(round(n / grid_h))

    if grid_h * grid_w != n:
        print(f"Skip: cannot reshape {n} into ({grid_h}, {grid_w})")
        return

    # reshape
    score_map = scores.view(grid_h, grid_w).cpu().numpy()

    # resize
    heatmap = cv2.resize(
        score_map,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # 转 uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # 彩色热力图
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_VIRIDIS
    )

    # overlay
    overlay = cv2.addWeighted(
        img,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    # ===== 生成 colorbar =====
    fig, ax = plt.subplots(figsize=(1.2, 6))

    norm = plt.Normalize(vmin=0, vmax=1)

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap='viridis'),
        cax=ax
    )

    cbar.set_label('Score')

    fig.tight_layout()

    # 保存临时 colorbar
    colorbar_path = os.path.join(output_image_dir, "_temp_colorbar.png")
    fig.savefig(
        colorbar_path,
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.close(fig)

    # 读取 colorbar
    colorbar_img = cv2.imread(colorbar_path)

    # resize 高度一致
    cb_h, cb_w = colorbar_img.shape[:2]

    new_cb_w = int(cb_w * h / cb_h)

    colorbar_img = cv2.resize(
        colorbar_img,
        (new_cb_w, h)
    )

    # 拼接
    final = np.concatenate(
        [overlay, colorbar_img],
        axis=1
    )

    # 保存
    save_path = os.path.join(
        output_image_dir,
        f"{step}.png"
    )

    cv2.imwrite(save_path, final)

    # 删除临时文件
    os.remove(colorbar_path)
    
def compute_mask_from_json(image_name: str, h: int, w: int):
    json_path = os.environ['layout_json_path']

    with open(json_path, "r") as f:
        data = json.load(f)

    # 找到对应图片
    target = None
    for item in data:
        if item["page_info"]["image_path"] == image_name:
            target = item
            break

    if target is None:
        raise ValueError(f"Image {image_name} not found in JSON")

    page_h = target["page_info"]["height"]
    page_w = target["page_info"]["width"]

    # 初始化 mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # 遍历所有框
    for det in target["layout_dets"]:
        # if det.get("ignore", False):
        #     continue
        # if det.get("text", 1) == '':
        #     continue

        poly = det["poly"]  # [x1,y1,x2,y2,...]

        # 转为 (N,2)
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)

        # 坐标缩放到目标 h,w
        pts[:, 0] = pts[:, 0] / page_w * w
        pts[:, 1] = pts[:, 1] / page_h * h

        pts = pts.astype(np.int32)

        # 填充多边形
        cv2.fillPoly(mask, [pts], 1)

    return mask.astype(bool)
def save_score_mask(mask, scores, image_path, output_image_dir, threshold=.1):
    """
    scores: 1D torch.Tensor (0-1)
    image_path: str
    output_image_dir: str
    threshold: float，低于该值的区域变灰
    """
    os.makedirs(output_image_dir, exist_ok=True)

    # 读取图片
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # grid 总数
    n = scores.numel()

    # 推测 grid 行列数
    aspect = h / w
    grid_h = int(round(np.sqrt(n * aspect)))
    grid_w = int(round(n / grid_h))

    if grid_h * grid_w != n:
        print(f"mask: cannot reshape {n} into ({grid_h}, {grid_w})")
        return

    # reshape
    score_map = scores.view(grid_h, grid_w).cpu().numpy()

    # === 关键：生成mask ===
    mask = mask.view(grid_h, grid_w).numpy().astype(np.uint8)  # 低分区域=1

    # resize到原图大小
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 扩展为3通道
    mask_3c = np.stack([mask]*3, axis=-1)

    # === 构造灰色层 ===
    gray_layer = np.full_like(img, 128)  # 灰色 (0-255)

    # === 只在mask区域做blend ===
    alpha = 0.6  # 原图权重
    beta = 0.4   # 灰色权重

    overlay = img.copy()
    overlay[mask_3c == 1] = (
        img[mask_3c == 1] * alpha + gray_layer[mask_3c == 1] * beta
    ).astype(np.uint8)

    # 保存
    save_path = os.path.join(output_image_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, overlay)
    
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def save_score_mask_with_text(
    mask, 
    scores,
    image_path,
    output_image_dir,
    Levenshtein_ratio,
    gt,
    pred,
    threshold=0.1,
    font_path="layoutlite/utils/SimHei.ttf",  # 可换成中文字体
):
    """
    在mask图下方写入：
    - Levenshtein_ratio
    - gt
    - pred
    支持中文（需提供支持中文的字体）

    scores: 1D torch.Tensor (0-1)
    image_path: str
    output_image_dir: str
    """

    os.makedirs(output_image_dir, exist_ok=True)

    # === 读取图片 ===
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # === grid推断 ===
    n = scores.numel()
    aspect = h / w
    grid_h = int(round(np.sqrt(n * aspect)))
    grid_w = int(round(n / grid_h))

    if grid_h * grid_w != n:
        print(f"mask: cannot reshape {n} into ({grid_h}, {grid_w})")
        return

    score_map = scores.detach().float().view(grid_h, grid_w).cpu().numpy()

    # === mask ===
    
    mask = mask.view(grid_h, grid_w).numpy().astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_3c = np.stack([mask] * 3, axis=-1)

    gray_layer = np.full_like(img, 128)

    alpha, beta = 0.6, 0.4
    overlay = img.copy()
    overlay[mask_3c == 1] = (
        img[mask_3c == 1] * alpha + gray_layer[mask_3c == 1] * beta
    ).astype(np.uint8)

    # === 转 PIL 写文字 ===
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay_pil)

    # 字体（建议换成支持中文的，如 simhei.ttf / NotoSansCJK）
    try:
        font = ImageFont.truetype(font_path, 24)
    except:
        font = ImageFont.load_default()

    # === 文本内容 ===
    text_lines = [
        f"Levenshtein_ratio: {Levenshtein_ratio:.4f}",
        f"GT: {gt}",
        f"PRED: {pred}",
    ]

    # === 计算文本区域高度 ===
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in text_lines]
    padding = 10
    text_height = sum(line_heights) + padding * (len(text_lines) + 1)

    # === 创建新画布（在下方扩展）===
    new_img = Image.new("RGB", (w, h + text_height), (255, 255, 255))
    new_img.paste(overlay_pil, (0, 0))

    draw = ImageDraw.Draw(new_img)

    # === 写文字 ===
    y = h + padding
    for line, lh in zip(text_lines, line_heights):
        draw.text((10, y), line, fill=(0, 0, 0), font=font)
        y += lh + padding

    # === 保存 ===
    save_path = os.path.join(output_image_dir, os.path.basename(image_path))
    new_img.save(save_path)
    
PROMPT_LOGICS = "QwenVL HTML"
PROMPT_FIRERED = '''You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

            1. Text Processing:
            - Accurately recognize all text content in the PDF image without guessing or inferring.
            - Convert the recognized text into Markdown format.
            - Maintain the original document structure, including headings, paragraphs, lists, etc.

            2. Mathematical Formula Processing:
            - Convert all mathematical formulas to LaTeX format.
            - Enclose inline formulas with,(,). For example: This is an inline formula,( E = mc^2,)
            - Enclose block formulas with,\[,\]. For example:,[,frac{-b,pm,sqrt{b^2 - 4ac}}{2a},]

            3. Table Processing:
            - Convert tables to HTML format.
            - Wrap the entire table with <table> and </table>.

            4. Figure Handling:
            - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

            5. Output Format:
            - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
            - For complex layouts, try to maintain the original document's structure and format as closely as possible.

            Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
            '''
            