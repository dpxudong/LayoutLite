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

def save_score_mask(mask, scores, image_path, output_image_dir, threshold=.1):
    """
    scores: 1D torch.Tensor (0-1)
    image_path: str
    output_image_dir: str
    threshold: float，低于该值的区域变灰
    """
    os.makedirs(output_image_dir, exist_ok=True)
    mask = 1 - mask
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
    font_path="/data/houkaiji/doc_understanding_data_preprocess/SimHei.ttf",  # 可换成中文字体
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
    mask = 1 - mask
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
    
def save_score_heatmap(scores, image_path, output_image_dir):
    """
    scores: 1D torch.Tensor (0-1)
    image_path: str
    output_image_dir: str
    """
    import os
    import cv2
    import numpy as np

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
        print(f"heatmap: cannot reshape {n} into ({grid_h}, {grid_w})")
        return

    # reshape
    score_map = scores.view(grid_h, grid_w).cpu().numpy()

    # resize
    heatmap = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_NEAREST)
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    # ===== 在每个格子写数字 =====
    cell_h = h / grid_h
    cell_w = w / grid_w

    for i in range(grid_h):
        for j in range(grid_w):
            score = score_map[i, j]

            # 保留两位小数
            text = f"{score:.2f}"

            # 计算格子中心位置
            x = int((j + 0.5) * cell_w)
            y = int((i + 0.5) * cell_h)

            # cv2.putText(overlay, text, (x - 15, y + 5),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6, (0, 0, 0), 2, cv2.LINE_AA)


    # 保存
    save_path = os.path.join(output_image_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, overlay)