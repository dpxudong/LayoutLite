# model_type: firered/logics
infer_mode=full \
python scripts/train.py \
    --model_type logics \
    --model_dir /data/code/vlm_ocr_token_test/logics/Alibaba-DT/Logics-Parsing-v2 \
    --dataset /data/code/VL_RL/got_rl/dataset/OCRBench_v2.jsonl \
    --output_dir ./outputs \

