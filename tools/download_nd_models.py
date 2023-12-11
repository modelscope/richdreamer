from modelscope import snapshot_download

model_id = snapshot_download(
    "Damo_XR_Lab/Normal-Depth-Diffusion-Model", cache_dir="./pretrained_models"
)
