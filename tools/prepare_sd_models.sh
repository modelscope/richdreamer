# link your huggingface models to ./pretrained_models/huggingface
cd pretrained_models && ln -s ~/.cache/huggingface ./
cd -

# if you cannot visit huggingface to download SD 1.5 and SD 2.1,
# you can download SD models from [aliyun](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/models_sd.tar.gz)
# and then put `$download_sd` file to `pretrained_models/huggingface/hub/`.
mkdir -p pretrained_models/huggingface/hub/
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/models_sd_clip.tar.gz -O pretrained_models/models_sd_clip.tar.gz
mv pretrained_models/models_sd_clip.tar.gz pretrained_models/huggingface/hub/
cd pretrained_models/huggingface/hub/
tar -xvf models_sd_clip.tar.gz ./