from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="FireRedTeam/FireRed-OCR",
    local_dir="checkpoints/FireRed-OCR",
    local_dir_use_symlinks=False,
)