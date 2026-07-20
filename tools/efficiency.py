import os
import re
import json
import numpy as np

config_path = '/data/code/vlm_ocr_token_test/logics/Alibaba-DT/Logics-Parsing-v2/config.json'
folders = [
    # 'outputs/flops/firered0/eval_0715152425',
    # 'outputs/flops/firered10/eval_0715152530',
    # 'outputs/flops/firered20/eval_0715152536',
    # 'outputs/flops/firered30/eval_0715152545',
    # 'outputs/flops/firered40/eval_0715152554',
    # 'outputs/flops/firered50/eval_0715152606',
    # 'outputs/flops/firered60/eval_0715152617',
    # 'outputs/flops/firered70/eval_0715162903',
    '/data/code/VL_RL/LayoutLite/outputs/flops_logics/logics0/eval_0715174432',
    '/data/code/VL_RL/LayoutLite/outputs/flops_logics/logics10/eval_0715174604',
    '/data/code/VL_RL/LayoutLite/outputs/flops_logics/logics20/eval_0715174608',
    '/data/code/VL_RL/LayoutLite/outputs/flops_logics/logics30/eval_0715174615',
    '/data/code/VL_RL/LayoutLite/outputs/flops_logics/logics40/eval_0715174624',
    '/data/code/VL_RL/LayoutLite/outputs/flops_logics/logics50/eval_0715174633',
    '/data/code/VL_RL/LayoutLite/outputs/flops_logics/logics60/eval_0715174640',
    '/data/code/VL_RL/LayoutLite/outputs/flops_logics/logics70/eval_0716100708'
]

ttft_pattern = re.compile(r"ttft:\s*([0-9eE.+-]+)")
score_pattern = re.compile(r"score:\s*([0-9eE.+-]+)")
kv_pattern = re.compile(r"kv:\s*([0-9eE.+-]+)")
flops_pattern = re.compile(r"seq_len:\s*([0-9eE.+-]+)")

def calc_llm_prefill_tflops(input_len, config_path):
    """
    Calculate theoretical LLM Prefill FLOPs.

    Args:
        input_len (int):
            Number of input tokens entering the LLM (after token pruning).
        config_path (str):
            Path to config.json.

    Returns:
        float:
            Prefill FLOPs in TFLOPs.
    """

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Some HuggingFace models put LLM config under text_config
    tc = config["text_config"]

    L = tc["num_hidden_layers"]
    H = tc["hidden_size"]
    FFN = tc["intermediate_size"]
    nh = tc["num_attention_heads"]
    nkv = tc["num_key_value_heads"]
    dh = tc["head_dim"]

    S = input_len

    # Q/K/V/Out projections (GQA)
    flops_qkvo = (
        2
        * S
        * (
            H * nh * dh          # Q projection
            + H * nkv * dh * 2   # K + V projection
            + nh * dh * H        # Output projection
        )
    )

    # Attention score + weighted sum
    flops_attn = 4 * S * S * nh * dh

    # SwiGLU FFN (gate + up + down)
    flops_ffn = 2 * S * H * FFN * 3

    llm_prefill_tflops = (
        L * (flops_qkvo + flops_attn + flops_ffn) / 1e12
    )

    return llm_prefill_tflops

def calc_vit_prefill_tflops(vit_output_len, config_path):
    """
    Calculate theoretical Vision Transformer (ViT) FLOPs.

    Args:
        vit_output_lens (list of int):
            List containing the sequence length of vision tokens for each image.
        config_path (str):
            Path to config.json.

    Returns:
        float:
            ViT FLOPs in TFLOPs.
    """

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Some Multimodal models put Vision config under vision_config
    vc = config["vision_config"]

    # Extract model configurations with fallbacks
    _H = vc["hidden_size"]
    _FFN = vc["intermediate_size"]
    # ViT configs may use 'depth' or standard 'num_hidden_layers'
    _D = vc["depth"]

    # Calculate FLOPs for Linear layers per token per layer
    # - Projection matrix: QKV (6 * H^2) + Out (2 * H^2) = 8 * H^2
    # - MLP (non-gated): Up (2 * H * FFN) + Down (2 * H * FFN) = 4 * H * FFN
    _linear_per_tok = (4 * _H * _H + 2 * _H * _FFN) * 2

    # Calculate Attention self-attention matrix multiplication (QK^T and Attn_weights * V)
    # Total attention operations: 4 * N^2 * H FLOPs per layer for sequence length N
    _attn_tflops = _D * 4 * vit_output_len * vit_output_len * _H / 1e12

    # Total Vision FLOPs (Linear layers + Self-Attention layers)
    vit_flops_theory_tflops = (
        _D * vit_output_len * _linear_per_tok / 1e12 + _attn_tflops
    )

    return vit_flops_theory_tflops

all_ttft = []
all_score = []
all_kv = []
all_flops = []

for folder in folders:
    path = os.path.join(folder, "efficiency.txt")

    ttfts = []
    scores = []
    kvs = []
    flopss = []

    with open(path, "r") as f:
        for line in f:
            m1 = ttft_pattern.search(line)
            m2 = score_pattern.search(line)
            m3 = kv_pattern.search(line)
            m4 = flops_pattern.search(line)

            if m1:
                ttfts.append(float(m1.group(1)))
            if m2:
                scores.append(float(m2.group(1)))
            if m3:
                kvs.append(float(m3.group(1)))
            if m4:
                flopss.append(float(m4.group(1)))

    print(f"{folder}: ttft={len(ttfts)}, score={len(scores)}, kv={len(kvs)}")

    all_ttft.append(ttfts)
    all_score.append(scores)
    all_kv.append(kvs)
    all_flops.append(flopss)

# 统一统计长度
min_len = min(min(len(x) for x in all_ttft),
              min(len(x) for x in all_score),
              min(len(x) for x in all_kv),
              min(len(x) for x in all_flops))

print(f"\nUsing first {min_len} lines.\n")

for folder, ttfts, scores, kvs, flopss in zip(folders, all_ttft, all_score, all_kv, all_flops):
    mean_ttft = np.mean(ttfts[:min_len])
    mean_score = np.mean(scores[:min_len])
    mean_kv = np.mean(kvs[:min_len])
    mean_flops = np.mean(flopss[:min_len])

    print(
        f"{folder:<20}    "
        f"TTFT = {mean_ttft:.6f}    "
        f"Score = {mean_score:.6f}    "
        f"KV = {mean_kv:.6f}    "
        f"flops = {calc_llm_prefill_tflops(mean_flops, config_path) + calc_vit_prefill_tflops(np.mean(all_flops[0][:min_len]), config_path):.6f}"
    )
