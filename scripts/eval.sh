# model_type: firered/logics
python scripts/eval.py \
    --model_type firered \
    --model_dir /data/code/vlm_ocr_token_test/FireRed-OCR/FireRedTeam/FireRed-OCR \
    --layoutlite_ckp_path /data/code/VL_RL/FireRed/output_0602_firered/ckp_0602105631/ckp_600.pt \
    --dataset /data/code/VL_RL/got_rl/dataset/OmniDocBench_100%_md.jsonl \
    --output_dir ./outputs \
    --compression_ratio 0.5
    # --do_binary_search

