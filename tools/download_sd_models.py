from modelscope import snapshot_download

# runwayml/stable-diffusion-v1-5
# models--runwayml--stable-diffusion-v1-5
model_id = snapshot_download(
    "AI-ModelScope/stable-diffusion-v1.5-no-safetensor", cache_dir="./pretrained_models"
)
print(model_id)

# openai/clip-vit-large-patch14
# models--openai--clip-vit-large-patch14
model_id = snapshot_download(
    "AI-ModelScope/clip-vit-large-patch14", cache_dir="./pretrained_models"
)
print(model_id)

# stabilityai/stable-diffusion-2-1-base
# models--stabilityai--stable-diffusion-2-1-base
model_id = snapshot_download(
    "AI-ModelScope/stable-diffusion-2-1-base", cache_dir="./pretrained_models"
)
print(model_id)

# laion/CLIP-ViT-H-14-laion2B-s32B-b79K
# models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K
model_id = snapshot_download(
    "AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir="./pretrained_models"
)
print(model_id)