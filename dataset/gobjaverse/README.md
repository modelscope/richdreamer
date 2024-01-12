<p align="center">
    <h1>G-buffer Objaverse</h1>
<p>

G-buffer Objaverse: High-Quality Rendering Dataset of Objaverse.

[Chao Xu](mailto:eric.xc@alibaba-inc.com),
[Yuan Dong](mailto:yuandong15@fudan.edu.cn),
[Qi Zuo](mailto:muyuan.zq@alibaba-inc.com),
[Junfei Zhang](mailto:miracle.zjf@alibaba-inc.com),
[Xiaodan Ye](mailto:doris.yxd@alibaba-inc.com),
[Wenbo Geng](mailto:rengui.gwb@alibaba-inc.com),
[Yuxiang Zhang](mailto:yuxiangzhang.zyx@alibaba-inc.com),
[Xiaodong Gu](https://scholar.google.com.hk/citations?user=aJPO514AAAAJ&hl=zh-CN&oi=ao),
[Lingteng Qiu](https://lingtengqiu.github.io/),
[Zhengyi Zhao](mailto:bushe.zzy@alibaba-inc.com),
[Qing Ran](mailto:ranqing.rq@alibaba-inc.com),
[Jiayi Jiang](mailto:jiayi.jjy@alibaba-inc.com),
[Zilong Dong](https://scholar.google.com/citations?user=GHOQKCwAAAAJ&hl=zh-CN&oi=ao),
[Liefeng Bo](https://scholar.google.com/citations?user=FJwtMf0AAAAJ&hl=zh-CN)

## [Project page](https://aigc3d.github.io/gobjaverse/) | [YouTube](https://www.youtube.com/watch?v=PWweS-EPbJo) | [RichDreamer](https://aigc3d.github.io/richdreamer/) | [ND-Diffusion Model](https://github.com/modelscope/normal-depth-diffusion)

## News

- Release G-buffer Objaverse Rendering Dataset (01.06, 2024 UTC)
- Release 10 Category Annotation of the Objaverse Subset (01.06, 2024 UTC)
- Thanks for [JunzheJosephZhu](https://github.com/JunzheJosephZhu) for improving the robustness of the downloading scripts. Now you could restart the download script from the break point. (01.12, 2024 UTC)
- We will release a compressed version of the datasets, stay tuned!

## Download
- Download gobjaverse rendering dataset using following scripts.
```bash
# download_gobjaverse_280k index file
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/gobjaverse_280k.json
# Example: python ./scripts/data/download_gobjaverse_280k.py ./gobjaverse_280k ./gobjaverse_280k.json 10
python ./download_gobjaverse_280k.py /path/to/savedata /path/to/gobjaverse_280k.json nthreads(eg. 10)
# download Cap3D text-caption file
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/text_captions_cap3d.json

```
- The 10 general categories including Human-Shape (41,557), Animals (28,882), Daily-Used (220,222), Furnitures (19,284), Buildings&&Outdoor (116,545), Transportations (20,075), Plants (7,195), Food (5,314), Electronics (13,252) and Poor-quality (107,001).
- Download the category annotation using following scripts.

```bash
# download category annotation
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/category_annotation.json
```

## Folder Structure
- The structure of gobjaverse rendering dataset:
```
|-- ROOT
    |-- dictionary_id
        |-- instance_id
            |-- campos_512_v4
                |-- 00000
                    |-- 00000.json  # Camera Information
                    |-- 00000.png   # RGB 
                    |-- 00000_albedo.png  # Albedo 
                    |-- 00000_hdr.exr  # HDR
                    |-- 00000_mr.png  # Metalness and Roughness
                    |-- 00000_nd.exr  # Normal and Depth
                |-- ...
```

### Coordinate System

The 3D coordinate system definition is very complex. it is difficult  for us to say  what the camera system used. Fortunately, the target  we want to get is mapping the world normal of rendering system to Normal-Bae system,  as the following figure  illustrates:

![normal-bae system](./normal-bae system.png)

where the U-axis and V-axis denote the width-axis and height-axis in image space, respectively, the xyz is the Normal-Bae camera view coordinate system. 

Note that public rendering  system  for Objaverse is blender-based system, see:

![00000_normal](./blender_world_normal.png)

However, our rendering system is defined  at **Unity-based system** see:

![00000_normal](./unity-based.png)

*A question is how do we plug in blender's coordinate system directly without introducing a new coordinate system?*

A possible solution is that we maintain world to camera  transfer matrix as blender setting, *transferring Unity-based system to blender-based system*

We provide example codes to visualize the coordinate mapping.

```bash
# example of coordinate experiments
## download datasets
wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/Lingtengqiu/render_data_examples.zip
unzip render_data_examples.zip

## visualizing blender-based system, and warping world-space normal to normal-bae system.
python ./process_blender_dataset.py

## visualizing our system, and warping world-space normal to normal-bae system.

```



## Citation	

```
@article{qiu2023richdreamer,
    title={RichDreamer: A Generalizable Normal-Depth Diffusion Model for Detail Richness in Text-to-3D}, 
    author={Lingteng Qiu and Guanying Chen and Xiaodong Gu and Qi zuo and Mutian Xu and Yushuang Wu and Weihao Yuan and Zilong Dong and Liefeng Bo and Xiaoguang Han},
    year={2023},
    journal = {arXiv preprint arXiv:2311.16918}
}
```
```
@article{objaverse,
    title={Objaverse: A Universe of Annotated 3D Objects},
    author={Matt Deitke and Dustin Schwenk and Jordi Salvador and Luca Weihs and
            Oscar Michel and Eli VanderBilt and Ludwig Schmidt and
            Kiana Ehsani and Aniruddha Kembhavi and Ali Farhadi},
    journal={arXiv preprint arXiv:2212.08051},
    year={2022}
}
```
