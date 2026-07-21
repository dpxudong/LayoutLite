# model_type: firered/logics
infer_mode=full \
python scripts/train.py \
    --model_type firered \
    --model_dir checkpoints/Logics-Parsing-v2 \
    --dataset datasets/OCRBench_v2.jsonl \
    --output_dir ./outputs