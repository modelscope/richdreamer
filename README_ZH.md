<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
    <h1>RichDreamer</h1>
<p>

RichDreamer:一个基于通用的法向-深度扩散模型, 从文本生成细节丰富3D模型的新方法。

[Lingteng Qiu\*](https://lingtengqiu.github.io/),
[Guanying Chen\*](https://guanyingc.github.io/),
[Xiaodong Gu\*](https://scholar.google.com.hk/citations?user=aJPO514AAAAJ&hl=zh-CN&oi=ao),
Qi Zuo,
[Mutian Xu](https://mutianxu.github.io/),
Yushuang Wu,
[Weihao Yuan](https://weihao-yuan.com/),
[Zilong Dong](https://scholar.google.com/citations?user=GHOQKCwAAAAJ&hl=zh-CN&oi=ao),
[Liefeng Bo](https://research.cs.washington.edu/istc/lfb/),
[Xiaoguang Han](https://gaplab.cuhk.edu.cn/)

法向-深度扩散模型的更多细节请参阅[normal-depth-diffusion](https://github.com/modelscope/normal-depth-diffusion)。

## [项目主页](https://lingtengqiu.github.io/RichDreamer/)| [论文](https://arxiv.org/abs/2311.16918) | [bilibili](https://www.bilibili.com/video/BV1Qb4y1K7Sb/?spm_id_from=888.80997.embed_other.whitelist) | [法向-深度扩散模型](https://github.com/modelscope/normal-depth-diffusion)


<img src=".\figs\richdreamer.gif" alt="richdreamer" style="zoom:200%;" />  

## 待办事项 :triangular_flag_on_post:  
- [x]  文本到ND扩散模型  
- [x]  多视角ND和多视角反照率扩散模型  
- [x]  发布代码  
- [ ]  Docker 镜像  
- [ ]  在[ModelScope的3D物体生成](https://modelscope.cn/studios/Damo_XR_Lab/3D_AIGC/summary)上提供生成试用  

## 新闻  

- 发布 RichDreamer :fire::fire::fire:(UTC 2023年12月11日)  

## 架构  

![architecture](figs/architecture.png)  


# 安装  

- 系统要求:Ubuntu20.04  
- 测试GPU:RTX4090 或 A100  

使用以下脚本安装要求

```bash  
git clone https://github.com/modelscope/RichDreamer.git --recursive  
conda create -n rd  
conda activate rd  
# 安装threestudio的依赖
pip install -r requirements_3d.txt  
```  

我们还提供了dockerfile来构建docker镜像,或使用我们构建的[docker镜像](https://code.alibaba-inc.com/dadong.gxd/dream3d/blob/release/1209)。  

```bash
sudo docker build -t mv3dengine_22.04:cu118 -f docker/Dockerfile .  
```  

下载预训练权重

1. 多视角法向-深度扩散模型 [ND-MV](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/nd_mv_ema.ckpt)
2. 多视角深度图条件控制的反照率扩散模型 [Alebdo-MV](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/albedo_mv_ema.ckpt)

**或者**您可以使用以下脚本下载权重

```bash  
python tools/download_nd_models.py  
# 拷贝256_tets文件,供DMTet使用  
cp ./pretrained_models/Damo_XR_Lab/Normal-Depth-Diffusion-Model/256_tets.npz ./load/tets/  
# 将huggingface模型链接到./pretrained_models/huggingface  
cd pretrained_models && ln -s ~/.cache/huggingface ./  
```  

如果您无法访问huggingface下载SD 1.5，SD 2.1和CLIP的模型,您可以从[阿里云](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/RichDreamer/models_sd_clip.tar. gz)下载SD模型,然后将下载的的文件 `$download_sd_clip` 放入 `pretrained_models/huggingface/hub/`。  

```bash  
mkdir -p pretrained_models/huggingface/hub/  
cd pretrained_models/huggingface/hub/  
mv /path/to/${download_sd_clip} ./  
tar -xvf ${download_sd_clip} ./  
```  

## 3D生成  
确保你有以下的模型文件：
```bash
RichDreamer
|-- pretrained_models
    |-- Damo_XR_Lab
        |-- Normal-Depth-Diffusion-Model
            |-- nd_mv_ema.ckpt
            |-- albedo_mv_ema.ckpt
    
    |-- huggingface
        |-- hub
            |-- models--runwayml--stable-diffusion-v1-5
            |-- models--openai--clip-vit-large-patch14
            |-- models--stabilityai--stable-diffusion-2-1-base
            |-- models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K
```

请注意，在运行命令之前，我们在所有“*.sh”文件中设置环境变量“TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 HF_HUB_OFFLINE=1”，以防止每次连接到Huggingface。

如果你使用上面的脚本下载SD和CLIP模型，你什么都不用做，如果你通过huggingface api下载，`在第一次运行`时，你需要在`*.sh`中设置`TRANSFORMERS_OFFLINE=0 DIFFUSERS_OFFLINE=0 HF_HUB_OFFLINE=0`， 程序会连接Huggingface自动下载模型。

### 基于NeRF

```bash
# 快速启动,单A-100 80G  
python3 ./run_nerf.py -t $prompt -o $output  

# 使用文本列表批量运行  
# 例如：bash ./scripts/nerf/run_batch.sh 0 1 ./prompts_nerf.txt  
# 我们还提供了run_batch_res256.sh使用高分辨率渲染图像来优化以获得更好的效果，但它会消耗更多的内存和时间。
bash ./scripts/nerf/run_batch.sh $start_id $end_id ${prompts_nerf.txt}  

# 如果您没有A-100设备,我们提供了一个节省内存的版本来生成结果
# 比如单个GTX-3090/4090,24GB GPU内存
python3 ./run_nerf.py -t $prompt -o $output -s 1  
```  

### 基于DMTet  

#### DMTet训练提示  

**1. 渲染高分辨率:**  

我们发现, 与NeRF方法相比, 直接优化高分辨率DMTet球体更具挑战性。 例如,Fantasia3D和SweetDreamer都需要4或8个GPU进行优化, 这对大多数个人来说都很难获得。在实验过程中, 我们观察到, 当我们增加DMTet的渲染分辨率时, 例如 **1024**, 优化会变得显着更稳定。这种技巧使我们能够仅使用单个GPU从DMTet进行优化, 这在以前是不可行的。  

**2. PBR建模:**  

Fantasia3D提供了三种进行PBR建模的策略。 如果您**不**需要生成支持重新照明且仅目标增强实际效果的模型, 我们建议使用采样策略 *fantasia3d_2*。**否则**我们建议您使用我们的*深度条件反照率SDS*的*fantasia3d strategy_0*。  


```bash  
# 快速启动,单个A-100 80G  
python3 ./run_dmtet.py -t $prompt -o $output  

# 使用文本列表批量运行  
# 例如：bash ./scripts/nerf/run_batch.sh 0 1 ./prompts_dmtet.txt
bash ./scripts/dmtet/run_batch.sh $start_id $end_id ${prompt_dmtet.txt}   

# 如果您没有A-100设备,我们提供了一个节省内存的版本来生成结果
# 比如：单个GTX-3090/4090,24GB GPU内存  
# bash ./scripts/dmtet/run_batch_fast.sh 0 1 ./prompts_dmtet.txt  
bash ./scripts/dmtet/run_batch_fast.sh $start_id $end_id ${prompts_dmtet.txt}   
```  


## 致谢  

这项工作建立在许多惊人的研究工作和开源项目的基础上:  

- [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion)  
- [threestudio](https://github.com/threestudio-project/threestudio)  
- [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D)  
- [MVDream](https://github.com/bytedance/MVDream-threestudio)  

感谢他们出色的工作和对3D生成领域的巨大贡献。  

我们要特别感谢[Rui Chen](https://aruichen.github.io/)对Fantasia3D训练和PBR建模的宝贵讨论。   

此外, 我们衷心感谢Chao Xu在进行重光照实验方面的帮助。

## 引用      

```  
@article{qiu2023richdreamer,   
    title={RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D},    
    author={Lingteng Qiu and Guanying Chen and Xiaodong Gu and Qi zuo and Mutian Xu and Yushuang Wu and Weihao Yuan and Zilong Dong and Liefeng Bo and Xiaoguang Han},   
    year={2023},   
    journal = {arXiv preprint arXiv:2311.16918}  
}  
```