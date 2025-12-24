# ComfyUI_TRELLIS2_SM
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2):Native and Compact Structured Latents for 3D Generation,load single dinov3

# Update
* update texture mode support /同步官方的材质赋予模式，只需要一个ply 加贴图，就可以赋予PBR材质
* if use texture mode  need update json files ， 如果使用材质模式，主模型的json文件有几个是修改了的，注意同步，外置mesh接口 迟点更新
* 更适合中国宝宝的插件  make it comfy，

# 1. Installation

In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_TRELLIS2_SM.git
```
---

# 2. Requirements  
* 2.1 step 1   
```
pip install -r requirements.txt
```
* 2.2 step 2
* install (nvdiffrast,nvdiffrec,CuMesh,flex_gemm,o-voxel )  ,if failed to install ,find a wheel.. /不好安装就去找轮子吧
* [some wheel ](https://huggingface.co/smthem/TRELLIS.2-Wheels/tree/main)
```
export TORCH_CUDA_ARCH_LIST="8.9"   # dep your cuda list
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast --no-build-isolation

git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
pip install /tmp/extensions/nvdiffrec --no-build-isolation

git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
#pip install git+https://github.com/ashawkey/cubvh --no-build-isolation # if false  git clone --recursive https://github.com/ashawkey/cubvh
cd cubvh
# edit files ' /cubvh/include/gpu/bvh.cuh ' lines 29, annotation printf // 禁用警告输出，否则在多三角面的生成时 会刷新很多条警告
if (!m_overflowed) {
        // printf("WARNING TOO BIG (stack overflow)\n");  
    }
pip install . --no-build-isolation
 
# sudo apt install libeigen3-dev #linux

git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
pip install /tmp/extensions/FlexGEMM --no-build-isolation

cd ComfyUI/custom_nodes/ComfyUI_TRELLIS2_SM/TRELLIS_2
mkdir -p /tmp/extensions
cp -r o-voxel /tmp/extensions/o-voxel
pip install /tmp/extensions/o-voxel --no-build-isolation
```

  
# 3.Model
* 3.2 download TRELLIS2 checkpoints  from  [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B/tree/main)  从hg或者魔搭下载所有文件,文件结构如下图
* 3.2 download dinov2 checkpoints [ facebook/dinov3-vitl16-pretrain-lvd1689m](https://www.modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m) safetensors only / 只下载模型文件
* 3.3 download RMBG-2.0 checkpoints  [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0/tree/main) safetensors only / 只下载模型文件
```
--  ComfyUI/models/dinov2/
    |-- dinov3-vitl16-pretrain-lvd1689m.safetensors  # rename or not
    |-- RMBG.safetensors   # rename or not
--  any_path/TRELLIS.2-4B/
    |-- ckpts
        |-- all files  #所有文件  # if use texture mode  need update it 如果使用材质模式，模型的json文件有几个是修改了的，注意同步
    |-- pipeline.json
    |-- texturing_pipeline.json # texture mode 

```


# Example
* normal image2glb
![](https://github.com/smthemex/ComfyUI_TRELLIS2_SM/blob/main/example_workflows/example.png)
* ply+imamg,texture mode /PRB材质模式
![](https://github.com/smthemex/ComfyUI_TRELLIS2_SM/blob/main/example_workflows/example_t.png)

# Citation
```
@article{
    xiang2025trellis2,
    title={Native and Compact Structured Latents for 3D Generation},
    author={Xiang, Jianfeng and Chen, Xiaoxue and Xu, Sicheng and Wang, Ruicheng and Lv, Zelong and Deng, Yu and Zhu, Hongyuan and Dong, Yue and Zhao, Hao and Yuan, Nicholas Jing and Yang, Jiaolong},
    journal={Tech report},
    year={2025}
}

``
