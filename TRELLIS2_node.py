# !/usr/bin/env python

# -*- coding: UTF-8 -*-
import numpy as np
import os
import torch
from pathlib import PureWindowsPath
from .utils import glb2obj_,obj2fbx_,tensor2imglist,preprocess_image2alpha,image2alpha
import folder_paths
import random
import numpy as np
import comfy
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from .TRELLIS_2.trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor  
from transformers import AutoConfig
import re

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
current_path = os.path.dirname(os.path.abspath(__file__))

weigths_dinov2_current_path = os.path.join(folder_paths.models_dir, "dinov2")
if not os.path.exists(weigths_dinov2_current_path):
    os.makedirs(weigths_dinov2_current_path)
folder_paths.add_model_folder_path("dinov2", weigths_dinov2_current_path)


class Trellis2_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="Trellis2_SM_Model",
            display_name="Trellis2_SM_Model",
            category="Trellis2_SM",
            inputs=[
                io.String.Input("repo",multiline=False,default="microsoft/TRELLIS.2-4B"),
                io.Combo.Input("attn_backend",options= ["xformers","flash-attn"]),
                io.Combo.Input("spconv_algo",options= ["auto","flash-native"]),  
                io.Boolean.Input("texture_mode",optional=False),
            ],
            outputs=[
                io.Custom("Trellis2_SM_Model").Output("model"),
                ],
            )
    @classmethod
    def execute(cls, repo,attn_backend,spconv_algo, texture_mode) -> io.NodeOutput:
        if attn_backend=="xformers":
            os.environ['ATTN_BACKEND'] = 'xformers'
        else:
            os.environ['ATTN_BACKEND'] = 'flash-attn'
        if spconv_algo=="auto":
            os.environ['SPCONV_ALGO'] = 'auto'
        else:
            os.environ['SPCONV_ALGO'] = 'native'
        if repo:
            repo=PureWindowsPath(repo).as_posix()
        else:
            raise "need fill repo"  
        if texture_mode:
            from .TRELLIS_2.trellis2.pipelines import Trellis2TexturingPipeline
            model=Trellis2TexturingPipeline.from_pretrained(repo,config_file="texturing_pipeline.json")
            model.texture_mode=True
        else:
            from .TRELLIS_2.trellis2.pipelines import Trellis2ImageTo3DPipeline
            model=Trellis2ImageTo3DPipeline.from_pretrained(repo,config_file= "pipeline.json")
            model.texture_mode=False
        return io.NodeOutput(model)
    
class Trellis2_SM_Dino(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="Trellis2_SM_Dino",
            display_name="Trellis2_SM_Dino",
            category="Trellis2_SM",
            inputs=[
                io.Combo.Input("dino_clip",options= ["none"] + folder_paths.get_filename_list("dinov2")),],
            outputs=[
                io.Custom("Trellis2_SM_Dino").Output("dino"),
                ]
        )
    @classmethod
    def execute(cls, dino_clip) -> io.NodeOutput:
        dino_ckpt=folder_paths.get_full_path("dinov2", dino_clip) if dino_clip!="none" else None
        if dino_ckpt is None:
            repo="facebookresearch/dinov3-vitl16-pretrain-lvd1689m"
        else:
            repo=os.path.join(current_path, "facebookresearch/dinov3-vitl16-pretrain-lvd1689m") 
        model = DinoV3FeatureExtractor(repo,ckpt=dino_ckpt,)
        return io.NodeOutput(model)


class Trellis2_SM_Preprocess(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="Trellis2_SM_Preprocess",
            display_name="Trellis2_SM_Preprocess",
            category="Trellis2_SM",
            inputs=[
                io.Custom("Trellis2_SM_Dino").Input("model"),
                io.Image.Input("image"),# [B,H,W,C], C=3
                io.String.Input("mesh_path",default=""),
                io.Combo.Input("birefnet_ckpt",options= ["none"] + folder_paths.get_filename_list("dinov2")),
                io.Boolean.Input("low_vram",default=True),
                io.Mask.Input("mask",optional=True),    
                ],
            outputs=[
                io.Conditioning.Output(display_name="conds"),
                ],
            )
    @classmethod
    def execute(cls, model,image,mesh_path,birefnet_ckpt,low_vram,mask=None) -> io.NodeOutput:
        # preprocess image
        rmgb_ckpt_path=folder_paths.get_full_path("dinov2", birefnet_ckpt) if birefnet_ckpt!="none" else None
        if   mask is not None:
            if mask.shape[1]!=64 :
                image_list=image2alpha(image,mask)
            elif mask.shape[1]==64 and  rmgb_ckpt_path is not None:
                print("mask shape is [64 64] use birefnet，如果图片不带mask,则使用birefnet") 
                image_list=tensor2imglist(image) #normal iamge  
                print("Loading BriaRMBG 2.0 model...")
                from .facebookresearch.RMBG.birefnet import BiRefNet
                config = AutoConfig.from_pretrained(os.path.join(current_path,"facebookresearch/RMBG"), trust_remote_code=True)
                rembg_model = BiRefNet(True,config=config)
                sd=comfy.utils.load_torch_file(rmgb_ckpt_path)
                rembg_model.load_state_dict(sd,strict=False)
                del sd
                rembg_model.eval()    
                image_list=preprocess_image2alpha(image_list,device,rembg_model,False,low_vram)
            else:
                raise "mask shape must be [B,H,W] or HW ,[64 64] is not a useable mask shape，如果图片不带mask,则使用默认输出的mask是64" #如果图片不带mask,则使用默认输出的mask是64
                
        elif rmgb_ckpt_path is not None:
            image_list=tensor2imglist(image) #normal iamge  
            print("Loading BriaRMBG 2.0 model...")
            from .facebookresearch.RMBG.birefnet import BiRefNet
            config = AutoConfig.from_pretrained(os.path.join(current_path,"facebookresearch/RMBG"), trust_remote_code=True)
            rembg_model = BiRefNet(True,config=config)
            sd=comfy.utils.load_torch_file(rmgb_ckpt_path)
            rembg_model.load_state_dict(sd,strict=False)
            del sd
            rembg_model.eval()    
            image_list=preprocess_image2alpha(image_list,device,rembg_model,False,low_vram)
        else:
            raise "need link mask or birefnet_repo or birefnet_ckpt"
        model.to(device)
        cond_list=[]  
        for img in image_list:   
            cond_dict={}
            for image_size in [512,1024]:
                model.image_size = image_size          
                cond = model([img])
                neg_cond = torch.zeros_like(cond)
                conds={'cond': cond,'neg_cond': neg_cond,}
                cond_dict[image_size]=conds
            cond_list.append(cond_dict)
        if low_vram:
            model.to("cpu")

       
        if mesh_path!="":
            mesh_path=PureWindowsPath(mesh_path).as_posix()
            import trimesh
            if os.path.isfile(mesh_path):
                mesh_=trimesh.load(mesh_path)
                if isinstance(mesh_, trimesh.Scene):
                    mesh_ = trimesh.util.concatenate([
                            trimesh.Trimesh(vertices=g.vertices, faces=g.faces) 
                            for g in mesh_.geometry.values() 
                            if hasattr(g, 'vertices') and hasattr(g, 'faces')
                        ])
                    mesh_list = [mesh_]
                elif not isinstance(mesh_, trimesh.Trimesh):
                    raise ValueError(' not a supported mesh type. 输入文件不是有效的网格或场景')
                else:
                    mesh_list = [mesh_]
            elif os.path.isdir(mesh_path):
                ply_files = []
                for root, dirs, files in os.walk(mesh_path):
                    for file in files:
                        if file.lower().endswith('.ply') or file.lower().endswith('.obj') or file.lower().endswith('.glb'):
                            ply_files.append(os.path.abspath(os.path.join(root, file)))
                mesh_list=[]
                if ply_files:
                    for i in ply_files:
                        mesh = trimesh.load(i) 
                        if isinstance(mesh, trimesh.Scene):
                            geometries = list(mesh.geometry.values())
                            if geometries:
                                mesh = trimesh.util.concatenate([
                                    g if isinstance(g, trimesh.Trimesh) else trimesh.Trimesh(vertices=g.vertices, faces=getattr(g, 'faces', []))
                                    for g in geometries
                                ])
                        elif not isinstance(mesh, trimesh.Trimesh):
                            raise ValueError(' not a supported mesh type. 输入文件不是有效的网格或场景')
                        else:
                            pass
                        mesh_list.append(mesh) 
                else:
                    mesh_list=[]
                    print("目录中没有找到.ply文件")
            else:
                mesh_list=[]
                print ("mesh_path is not a file or directory")
        else:
            mesh_list=[]
        cond_dict={}
        cond_dict["cond_list"]=cond_list
        cond_dict["mesh_list"]=mesh_list
        return io.NodeOutput(cond_dict)


class Trellis2_SM_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="Trellis2_SM_Sampler",
            display_name="Trellis2_SM_Sampler",
            category="Trellis2_SM",
            inputs=[
                io.Custom("Trellis2_SM_Model").Input("model"),
                io.Conditioning.Input("conds"),#{ }
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Int.Input("texture_size", default=4096, min=512, max=MAX_SEED, step=512, display_mode=io.NumberDisplay.number),
                io.Combo.Input("pipeline_type",options=['1024_cascade','1024','512','1536_cascade']),
                io.Boolean.Input("remesh",default=True),
            ],
            outputs=[
                io.String.Output(display_name="model_path"),
                ],
        )
    @classmethod
    def execute(cls, model,conds,seed,texture_size,pipeline_type,remesh) -> io.NodeOutput: # 暂时按默认参数跑 use default parameters
        try:
            import o_voxel
            use_voxel = True
        except ImportError:
            use_voxel = False
            print("o_voxel not install, output none ") 
        output_path = []
        model.to(device)
        cond_list=conds["cond_list"]
        mesh_list=conds["mesh_list"]
        if model.texture_mode and not mesh_list:
            raise("texture_mode is True, but mesh_list is empty,使用材质模式需要填入ply目录或者文件路径")
        if not model.texture_mode and mesh_list:
            print("texture_mode is False, but mesh_list is not empty,使用的常规模式，但是有mesh输入，忽略mesh输入")
        if not model.texture_mode:
            for cond in cond_list:
                mesh = model.run(cond,seed=seed,pipeline_type=pipeline_type)[0] #default 1024_cascade
                mesh.simplify(16777216) # nvdiffrast limit
                prefix = ''.join(random.choice("0123456789") for _ in range(5))
                glb_path = f"{folder_paths.get_output_directory()}/Trellis2_{prefix}.glb"
                if use_voxel: 
                    glb = o_voxel.postprocess.to_glb(
                        vertices            =   mesh.vertices,
                        faces               =   mesh.faces,
                        attr_volume         =   mesh.attrs,
                        coords              =   mesh.coords,
                        attr_layout         =   mesh.layout,
                        voxel_size          =   mesh.voxel_size,
                        aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        decimation_target   =   1000000,
                        texture_size        =   texture_size,
                        remesh              =   remesh,
                        remesh_band         =   1,
                        remesh_project      =   0,
                        verbose             =   True
                    )
                    glb.export(glb_path, extension_webp=True)
                output_path.append(glb_path)
        else:
            resolution = int(re.search(r'\d+', pipeline_type).group())
            for cond,mesh in zip(cond_list,mesh_list):
                prefix = ''.join(random.choice("0123456789") for _ in range(5))
                glb_path = f"{folder_paths.get_output_directory()}/Trellis2_{prefix}_texture.glb"
                output = model.run(mesh,cond,seed=seed,resolution=resolution,) 
                output.export(glb_path, extension_webp=True)
                output_path.append(glb_path)           

        model_path = '\n'.join(output_path)
        model.to("cpu")
        torch.cuda.empty_cache()
        return io.NodeOutput(model_path)


class Trellis2_SM_Save(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="Trellis2_SM_Save",
            display_name="Trellis2_SM_Save",
            category="Trellis2_SM",
            inputs=[
                io.String.Input("model_path",force_input=True),#{ }
                io.Boolean.Input("glb2obj",default=False),
                io.Boolean.Input("glb2fbx",default=False),
                    # HW or BHW
            ],
            outputs=[
                io.String.Output(display_name="model_path"),
                ],
        )
    @classmethod
    def execute(cls, model_path,glb2obj,glb2fbx,) -> io.NodeOutput:
        output_path =  [line for line in model_path.splitlines() if line.strip()]
        if glb2obj:
            obj_paths=[]
            for path in output_path:
                obj_path=os.path.join(os.path.split(path)[0],os.path.split(path)[1].replace(".glb",".obj"))
                glb2obj_(path, obj_path)
                obj_paths.append(obj_path)
            if glb2fbx:
                fbx_paths=[]
                for i in obj_paths:
                    fbx_path = os.path.join(os.path.split(i)[0], os.path.split(i)[1].replace(".obj", ".fbx"))
                    obj2fbx_(i, fbx_path)
                    fbx_paths.append(fbx_path)
                output_path=fbx_paths
            else:
                output_path=obj_paths
        else:
            if glb2fbx:
                obj_paths = []
                fbx_paths = []
                for path in output_path:
                    obj_path = os.path.join(os.path.split(path)[0], os.path.split(path)[1].replace(".glb", ".obj"))
                    glb2obj_(path, obj_path)
                    obj_paths.append(obj_path)
                for i in obj_paths:
                    fbx_path = os.path.join(os.path.split(i)[0], os.path.split(i)[1].replace(".obj", ".fbx"))
                    obj2fbx_(i, fbx_path)
                    fbx_paths.append(fbx_path)
                output_path = obj_paths
        model_path = '\n'.join(output_path)  
        return io.NodeOutput(model_path)


from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/Trellis2_SM_Extension")
async def get_hello(request):
    return web.json_response("Trellis2_SM_Extension")

class Trellis2_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Trellis2_SM_Model,
            Trellis2_SM_Dino,
            Trellis2_SM_Preprocess,
            Trellis2_SM_Sampler,
            Trellis2_SM_Save,
        ]
async def comfy_entrypoint() -> Trellis2_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return Trellis2_SM_Extension()


