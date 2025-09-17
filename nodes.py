"""
ComfyUI Aliyun Video Generation Nodes
阿里云视频生成节点
"""

import os
import json
import base64
import requests
import time
import random
from typing import Dict, Any, Optional, Tuple
import folder_paths
import numpy as np
from PIL import Image
import torch

class AliyunAPIKey:
    """阿里云API密钥配置节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的DASHSCOPE_API_KEY"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "get_api_key"
    CATEGORY = "Aliyun Video"
    
    def get_api_key(self, api_key: str) -> tuple:
        """返回API密钥"""
        if not api_key.strip():
            raise ValueError("API密钥不能为空")
        return (api_key.strip(),)


class AliyunVideoBase:
    """阿里云视频生成基类"""
    
    def __init__(self):
        # 不在初始化时设置API密钥，而是在调用时设置
        self.api_key = None
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        self.headers = None
    
    def set_api_key(self, api_key: str):
        """设置API密钥"""
        if not api_key or not api_key.strip():
            raise ValueError("API密钥不能为空")
        
        self.api_key = api_key.strip()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable"
        }
    
    def image_to_base64(self, image_tensor: torch.Tensor) -> str:
        """将图像张量转换为base64编码"""
        # 转换张量格式
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.squeeze(0)
        
        # 转换为PIL图像
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        
        # 转换为base64
        import io
        buffer = io.BytesIO()
        image_pil.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}"
    
    def create_task(self, payload: Dict[str, Any]) -> str:
        """创建视频生成任务"""
        if not self.headers:
            raise Exception("请先设置API密钥")
            
        response = requests.post(self.base_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")
        
        result = response.json()
        if result.get('code'):
            raise Exception(f"API错误: {result.get('code')} - {result.get('message')}")
        
        return result['output']['task_id']
    
    def wait_for_completion(self, task_id: str, timeout: int = 300) -> str:
        """等待任务完成并返回视频URL"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 查询任务状态 - 使用GET方法查询任务
            query_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            
            query_headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(query_url, headers=query_headers)
            
            if response.status_code == 200:
                result = response.json()
                status = result['output']['task_status']
                
                if status == 'SUCCEEDED':
                    return result['output']['video_url']
                elif status == 'FAILED':
                    raise Exception(f"视频生成失败: {result['output'].get('message', '未知错误')}")
                elif status in ['PENDING', 'RUNNING']:
                    print(f"任务状态: {status}, 等待中...")
                    time.sleep(10)
                else:
                    raise Exception(f"未知任务状态: {status}")
            else:
                print(f"查询任务状态失败: {response.status_code} - {response.text}")
                time.sleep(10)
        
        raise Exception(f"任务超时 ({timeout}秒)")
    
    def download_video(self, video_url: str) -> str:
        """下载视频到本地"""
        # 创建输出目录
        output_dir = folder_paths.get_output_directory()
        video_filename = f"aliyun_video_{int(time.time())}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        
        # 下载视频
        response = requests.get(video_url, stream=True)
        if response.status_code == 200:
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return video_path
        else:
            raise Exception(f"下载视频失败: {response.status_code}")


class AliyunTextToVideo(AliyunVideoBase):
    """阿里云文生视频节点"""
    
    # 中文到英文的模型映射
    MODEL_MAPPING = {
        "万相2.2-文生视频-增强版": "wan2.2-t2v-plus",
        "万相2.1-文生视频-快速版": "wanx2.1-t2v-turbo",
        "万相2.1-文生视频-增强版": "wanx2.1-t2v-plus"
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "forceInput": True
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "一只小猫在月光下奔跑"
                }),
                "model": (["万相2.2-文生视频-增强版", "万相2.1-文生视频-快速版", "万相2.1-文生视频-增强版"], {
                    "default": "万相2.2-文生视频-增强版"
                }),
                "size": (["1080*1920", "1920*1080", "1440*1440", "1632*1248", "1248*1632", "480*832", "832*480", "624*624"], {
                    "default": "1920*1080"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "duration": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10
                }),
                "prompt_extend": ("BOOLEAN", {
                    "default": True,
                    "label_on": "开启智能扩写",
                    "label_off": "关闭智能扩写"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "label_on": "显示水印",
                    "label_off": "隐藏水印"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "tooltip": "随机种子，-1为随机生成"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate_video"
    CATEGORY = "Aliyun Video"
    
    def generate_video(self, api_key: str, prompt: str, model: str, size: str, 
                      negative_prompt: str = "", duration: int = 5, 
                      prompt_extend: bool = True, watermark: bool = False, seed: int = -1) -> Tuple[str]:
        """生成文生视频"""
        # 设置API密钥
        self.set_api_key(api_key)
        
        # 将中文模型名称转换为英文
        english_model = self.MODEL_MAPPING.get(model, model)
        
        # 处理随机种子
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        payload = {
            "model": english_model,
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "size": size,
                "duration": duration,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
                "seed": seed
            }
        }
        
        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt
        
        print(f"开始生成视频: {prompt}")
        task_id = self.create_task(payload)
        print(f"任务ID: {task_id}")
        
        video_url = self.wait_for_completion(task_id)
        print(f"视频生成完成: {video_url}")
        
        video_path = self.download_video(video_url)
        print(f"视频已保存到: {video_path}")
        
        return (video_path,)


class AliyunImageToVideo(AliyunVideoBase):
    """阿里云图生视频节点"""
    
    # 中文到英文的模型映射
    MODEL_MAPPING = {
        "万相2.2-图生视频-增强版": "wan2.2-i2v-plus",
        "万相2.2-图生视频-快速版": "wan2.2-i2v-flash", 
        "万相2.1-图生视频-快速版": "wanx2.1-i2v-turbo",
        "万相2.1-图生视频-增强版": "wanx2.1-i2v-plus"
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "forceInput": True
                }),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "让图像中的内容动起来"
                }),
                "model": (["万相2.2-图生视频-增强版", "万相2.2-图生视频-快速版", "万相2.1-图生视频-快速版", "万相2.1-图生视频-增强版"], {
                    "default": "万相2.2-图生视频-增强版"
                }),
                "resolution": (["480P", "720P", "1080P"], {
                    "default": "720P"
                }),
            },
            "optional": {
                "prompt_extend": ("BOOLEAN", {
                    "default": True,
                    "label_on": "开启智能扩写",
                    "label_off": "关闭智能扩写"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "label_on": "显示水印",
                    "label_off": "隐藏水印"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "tooltip": "随机种子，-1为随机生成"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate_video"
    CATEGORY = "Aliyun Video"
    
    def generate_video(self, api_key: str, image: torch.Tensor, prompt: str, model: str, 
                      resolution: str, prompt_extend: bool = True, watermark: bool = False, seed: int = -1) -> Tuple[str]:
        """生成图生视频"""
        # 设置API密钥
        self.set_api_key(api_key)
        
        # 转换图像为base64
        image_base64 = self.image_to_base64(image)
        
        # 将中文模型名称转换为英文
        english_model = self.MODEL_MAPPING.get(model, model)
        
        # 处理随机种子
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        payload = {
            "model": english_model,
            "input": {
                "img_url": image_base64,
                "prompt": prompt
            },
            "parameters": {
                "resolution": resolution,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
                "seed": seed
            }
        }
        
        print(f"开始生成图生视频: {prompt}")
        task_id = self.create_task(payload)
        print(f"任务ID: {task_id}")
        
        video_url = self.wait_for_completion(task_id)
        print(f"视频生成完成: {video_url}")
        
        video_path = self.download_video(video_url)
        print(f"视频已保存到: {video_path}")
        
        return (video_path,)


class AliyunFirstLastFrameToVideo(AliyunVideoBase):
    """阿里云首尾帧生视频节点"""
    
    # 中文到英文的模型映射
    MODEL_MAPPING = {
        "万相2.1-首尾帧生视频-增强版": "wanx2.1-kf2v-plus"
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "forceInput": True
                }),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "从第一帧到最后一帧的平滑过渡"
                }),
                "model": (["万相2.1-首尾帧生视频-增强版"], {
                    "default": "万相2.1-首尾帧生视频-增强版"
                }),
                "resolution": (["720P"], {
                    "default": "720P"
                }),
            },
            "optional": {
                "prompt_extend": ("BOOLEAN", {
                    "default": True,
                    "label_on": "开启智能扩写",
                    "label_off": "关闭智能扩写"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "label_on": "显示水印",
                    "label_off": "隐藏水印"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "tooltip": "随机种子，-1为随机生成"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate_video"
    CATEGORY = "Aliyun Video"
    
    def generate_video(self, api_key: str, first_frame: torch.Tensor, last_frame: torch.Tensor, 
                      prompt: str, model: str, resolution: str, 
                      prompt_extend: bool = True, watermark: bool = False, seed: int = -1) -> Tuple[str]:
        """生成首尾帧视频"""
        # 设置API密钥
        self.set_api_key(api_key)
        
        # 转换图像为base64
        first_frame_base64 = self.image_to_base64(first_frame)
        last_frame_base64 = self.image_to_base64(last_frame)
        
        # 将中文模型名称转换为英文
        english_model = self.MODEL_MAPPING.get(model, model)
        
        # 处理随机种子
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        payload = {
            "model": english_model,
            "input": {
                "first_frame_url": first_frame_base64,
                "last_frame_url": last_frame_base64,
                "prompt": prompt
            },
            "parameters": {
                "resolution": resolution,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
                "seed": seed
            }
        }
        
        print(f"开始生成首尾帧视频: {prompt}")
        task_id = self.create_task(payload)
        print(f"任务ID: {task_id}")
        
        video_url = self.wait_for_completion(task_id)
        print(f"视频生成完成: {video_url}")
        
        video_path = self.download_video(video_url)
        print(f"视频已保存到: {video_path}")
        
        return (video_path,)


class AliyunVideoEffects(AliyunVideoBase):
    """阿里云视频特效节点"""
    
    # 中文到英文的模板映射
    TEMPLATE_MAPPING = {
        # 通用特效
        "解压捏捏": "squish",
        "转圈圈": "rotation", 
        "戳戳乐": "poke",
        "气球膨胀": "inflate",
        "分子扩散": "dissolve",
        # 单人特效
        "时光木马": "carousel",
        "爱你哟": "singleheart",
        "摇摆时刻": "dance1",
        "头号甩舞": "dance2", 
        "星摇时刻": "dance3",
        "人鱼觉醒": "mermaid",
        "学术加冕": "graduation",
        "巨兽追袭": "dragon",
        "财从天降": "money",
        # 单人或动物特效
        "魔法悬浮": "flying",
        "赠人玫瑰": "rose",
        "闪亮玫瑰": "crystalrose",
        # 双人特效
        "爱的抱抱": "hug",
        "唇齿相依": "frenchkiss",
        "双倍心动": "coupleheart"
    }
    
    # 中文到英文的模型映射
    MODEL_MAPPING = {
        "万相2.1-图生视频-快速版": "wanx2.1-i2v-turbo",
        "万相2.1-图生视频-增强版": "wanx2.1-i2v-plus"
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "forceInput": True
                }),
                "image": ("IMAGE",),
                "template": ([
                    # 通用特效
                    "解压捏捏", "转圈圈", "戳戳乐", "气球膨胀", "分子扩散",
                    # 单人特效
                    "时光木马", "爱你哟", "摇摆时刻", "头号甩舞", "星摇时刻", 
                    "人鱼觉醒", "学术加冕", "巨兽追袭", "财从天降",
                    # 单人或动物特效
                    "魔法悬浮", "赠人玫瑰", "闪亮玫瑰",
                    # 双人特效
                    "爱的抱抱", "唇齿相依", "双倍心动"
                ], {
                    "default": "魔法悬浮"
                }),
                "model": (["万相2.1-图生视频-快速版", "万相2.1-图生视频-增强版"], {
                    "default": "万相2.1-图生视频-快速版"
                }),
                "resolution": (["480P", "720P"], {
                    "default": "720P"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "tooltip": "随机种子，-1为随机生成"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate_video"
    CATEGORY = "Aliyun Video"
    
    def generate_video(self, api_key: str, image: torch.Tensor, template: str, 
                      model: str, resolution: str, seed: int = -1) -> Tuple[str]:
        """生成视频特效"""
        # 设置API密钥
        self.set_api_key(api_key)
        
        # 转换图像为base64
        image_base64 = self.image_to_base64(image)
        
        # 将中文模板名称转换为英文
        english_template = self.TEMPLATE_MAPPING.get(template, template)
        # 将中文模型名称转换为英文
        english_model = self.MODEL_MAPPING.get(model, model)
        
        # 处理随机种子
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        payload = {
            "model": english_model,
            "input": {
                "img_url": image_base64,
                "template": english_template
            },
            "parameters": {
                "resolution": resolution,
                "seed": seed
            }
        }
        
        print(f"开始生成视频特效: {template}")
        task_id = self.create_task(payload)
        print(f"任务ID: {task_id}")
        
        video_url = self.wait_for_completion(task_id)
        print(f"视频生成完成: {video_url}")
        
        video_path = self.download_video(video_url)
        print(f"视频已保存到: {video_path}")
        
        return (video_path,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AliyunAPIKey": AliyunAPIKey,
    "AliyunTextToVideo": AliyunTextToVideo,
    "AliyunImageToVideo": AliyunImageToVideo,
    "AliyunFirstLastFrameToVideo": AliyunFirstLastFrameToVideo,
    "AliyunVideoEffects": AliyunVideoEffects,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AliyunAPIKey": "阿里云API密钥",
    "AliyunTextToVideo": "阿里云文生视频",
    "AliyunImageToVideo": "阿里云图生视频", 
    "AliyunFirstLastFrameToVideo": "阿里云首尾帧生视频",
    "AliyunVideoEffects": "阿里云视频特效",
}