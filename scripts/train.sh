# model_type: firered/logics
infer_mode=full \
python scripts/train.py \
    --model_type firered \
    --model_dir checkpoints/FireRed-OCR \
    --dataset datasets/OCRBench_FireRed.jsonl \
    --output_dir ./outputs \
    --layout_json_path datasets/OCRBench_layout.json # Optional