from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Logics-MLLM/Logics-Parsing-v2",
    local_dir="checkpoints/Logics-Parsing-v2",
    local_dir_use_symlinks=False,
)