# --model_type: firered/logics
python scripts/eval.py \
    --model_type firered \
    --model_dir checkpoints/FireRed-OCR \
    --layoutlite_ckp_path checkpoints/firered_600.pt \
    --dataset datasets/OmniDocBench.jsonl \
    --output_dir ./outputs/test \
    --compression_ratio 0.5 \
    --layoutlite_scores_dir /path/to/layoutlite_scores # Optional. Provide calculated scores to skip LayoutLite execution.