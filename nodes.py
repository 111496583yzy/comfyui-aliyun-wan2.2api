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
import sys
import torch
import mimetypes
import uuid

def print_progress_bar(current: int, total: int, prefix: str = '', suffix: str = '', length: int = 50):
    """打印进度条"""
    percent = ("{0:.1f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if current == total:
        print()

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
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/video-synthesis"
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
        
        # 确保张量格式为 (C, H, W)
        if image_tensor.shape[0] == 3:  # RGB格式
            image_tensor = image_tensor.permute(1, 2, 0)  # 转换为 (H, W, C)
        
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
    
    def wait_for_completion(self, task_id: str, timeout: int = 300, model: str = None) -> str:
        """等待任务完成并返回视频URL"""
        # 根据模型类型设置不同的超时时间
        if model and "wanx2.1" in model:
            timeout = 900  # 2.1模型需要更长时间，设置为15分钟
            print(f"检测到2.1模型，超时时间设置为{timeout}秒")
        elif model and "wan2.2" in model:
            timeout = 300  # 2.2模型保持5分钟
            print(f"检测到2.2模型，超时时间设置为{timeout}秒")
        
        start_time = time.time()
        poll_interval = 15  # 根据官方文档建议，轮询间隔设置为15秒
        
        print(f"开始等待任务完成，超时时间: {timeout}秒，轮询间隔: {poll_interval}秒")
        
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
                elapsed_time = int(time.time() - start_time)
                
                # 检查是否有video_url字段，即使状态还是RUNNING
                video_url = result['output'].get('video_url')
                
                if status == 'SUCCEEDED' or video_url:
                    if video_url and status != 'SUCCEEDED':
                        print(f"检测到video_url，任务实际已完成但状态未更新")
                    print_progress_bar(100, 100, prefix='任务完成', suffix=f'耗时:{elapsed_time}s')
                    print(f"任务成功完成，耗时: {elapsed_time}秒")
                    return video_url
                elif status == 'FAILED':
                    error_msg = result['output'].get('message', '未知错误')
                    print(f"任务失败: {error_msg}")
                    raise Exception(f"视频生成失败: {error_msg}")
                elif status in ['PENDING', 'RUNNING']:
                    remaining_time = timeout - elapsed_time
                    progress_percent = min(95, (elapsed_time / timeout) * 100)  # 最多显示95%，避免过早完成
                    
                    if status == 'PENDING':
                        status_text = "排队中"
                    else:
                        status_text = "处理中"
                    
                    # 对于2.1模型，添加额外的状态检查信息
                    if model and "wanx2.1" in model:
                        print(f"2.1模型状态检查: {status}, video_url: {'存在' if video_url else '不存在'}")
                    
                    print_progress_bar(
                        int(progress_percent), 
                        100, 
                        prefix=f'任务{status_text}', 
                        suffix=f'{elapsed_time}s/{timeout}s 剩余:{remaining_time}s'
                    )
                    time.sleep(poll_interval)
                else:
                    print(f"未知任务状态: {status}")
                    raise Exception(f"未知任务状态: {status}")
            else:
                print(f"查询任务状态失败: {response.status_code} - {response.text}")
                time.sleep(poll_interval)
        
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
        print(f"使用模型: {english_model}")
        task_id = self.create_task(payload)
        print(f"任务ID: {task_id}")
        
        video_url = self.wait_for_completion(task_id, model=english_model)
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
        print(f"使用模型: {english_model}")
        task_id = self.create_task(payload)
        print(f"任务ID: {task_id}")
        
        video_url = self.wait_for_completion(task_id, model=english_model)
        print(f"视频生成完成: {video_url}")
        
        video_path = self.download_video(video_url)
        print(f"视频已保存到: {video_path}")
        
        return (video_path,)


class AliyunFirstLastFrameToVideo(AliyunVideoBase):
    """阿里云首尾帧生视频节点"""
    
    def __init__(self):
        super().__init__()
        # 首尾帧生视频使用相同的API端点
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/video-synthesis"
    
    # 中文到英文的模型映射
    MODEL_MAPPING = {
        "万相2.2-首尾帧生视频-极速版": "wan2.2-kf2v-flash",
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
                "model": (["万相2.2-首尾帧生视频-极速版", "万相2.1-首尾帧生视频-增强版"], {
                    "default": "万相2.2-首尾帧生视频-极速版"
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
        
        # 验证分辨率与模型的兼容性
        if english_model == "wanx2.1-kf2v-plus" and resolution != "720P":
            print(f"警告: {model} 只支持720P分辨率，已自动调整为720P")
            resolution = "720P"
        
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
        print(f"使用模型: {english_model}, 分辨率: {resolution}")
        task_id = self.create_task(payload)
        print(f"任务ID: {task_id}")
        
        video_url = self.wait_for_completion(task_id, model=english_model)
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
        print(f"使用模板: {english_template}")
        task_id = self.create_task(payload)
        print(f"任务ID: {task_id}")
        
        video_url = self.wait_for_completion(task_id, model="wan2.2-kf2v-flash")
        print(f"视频生成完成: {video_url}")
        
        video_path = self.download_video(video_url)
        print(f"视频已保存到: {video_path}")
        
        return (video_path,)


class FileUploadNode:
    """文件上传节点 - 支持上传到ComfyUI文件服务器"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入文件路径或拖拽文件到这里"
                }),
            },
            "optional": {
                "upload_to_server": ("BOOLEAN", {"default": True}),
                "server_url": ("STRING", {
                    "default": "https://ai.comfly.chat",
                    "multiline": False,
                    "placeholder": "上传服务器URL"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_url", "file_info")
    FUNCTION = "upload_file"
    CATEGORY = "Aliyun Video"
    
    def upload_file(self, file_path: str, upload_to_server: bool = True, server_url: str = "https://ai.comfly.chat") -> Tuple[str, str]:
        """上传文件并返回URL"""
        if not file_path or not os.path.exists(file_path):
            raise Exception(f"文件不存在: {file_path}")
        
        # 获取文件信息
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # 检查文件类型
        allowed_image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        allowed_video_exts = ['.mp4', '.avi', '.mov']
        
        if file_ext not in allowed_image_exts + allowed_video_exts:
            raise Exception(f"不支持的文件类型: {file_ext}")
        
        if upload_to_server and server_url:
            try:
                # 上传到指定服务器
                upload_url = f"{server_url}/v1/files"
                
                with open(file_path, 'rb') as file:
                    files = {
                        'file': (file_name, file, mimetypes.guess_type(file_path)[0] or 'application/octet-stream')
                    }
                    
                    # 发送上传请求
                    response = requests.post(upload_url, files=files, timeout=300)
                    
                    if response.status_code == 200:
                        result = response.json()
                        file_url = result.get('url', result.get('id'))
                        if not file_url:
                            raise Exception("服务器未返回文件URL")
                        
                        file_info = f"上传成功: {file_name}, 大小: {file_size} bytes, 类型: {file_ext}"
                        print(f"文件上传成功: {file_url}")
                        return (file_url, file_info)
                    else:
                        raise Exception(f"上传失败: {response.status_code} - {response.text}")
                        
            except Exception as e:
                print(f"上传到服务器失败: {e}")
                print("使用本地文件路径")
                # 上传失败，使用本地路径
                file_url = f"file://{file_path}"
                file_info = f"本地文件: {file_name}, 大小: {file_size} bytes (上传失败: {e})"
        else:
            # 返回本地文件路径
            file_url = file_path
            file_info = f"本地文件: {file_name}, 大小: {file_size} bytes"
        
        print(f"文件处理完成: {file_url}")
        return (file_url, file_info)


class AliyunImageToAnimateMove(AliyunVideoBase):
    """阿里云万相2.2图生动作节点"""
    
    # 服务模式说明
    MODE_DESCRIPTIONS = {
        "wan-std": "标准模式 - 生成速度快，满足基础动画演示等轻需求，性价比高 (0.4元/秒)",
        "wan-pro": "专业模式 - 动画流畅度高，动作表情过渡自然，效果更接近真实拍摄 (0.6元/秒)"
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "请输入您的DASHSCOPE_API_KEY"
                }),
                "image": ("IMAGE",),
                "video": (["STRING", "VIDEO"], {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入视频文件路径或连接LoadVideo节点"
                }),
                "mode": (["wan-std", "wan-pro"], {
                    "default": "wan-std"
                }),
                "check_image": ("BOOLEAN", {
                    "default": True
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "usage_info")
    FUNCTION = "generate_animate_move"
    CATEGORY = "Aliyun Video"
    
    def upload_image_to_server(self, image_tensor: torch.Tensor) -> str:
        """将图像上传到服务器并返回URL"""
        # 转换图像为base64
        image_base64 = self.image_to_base64(image_tensor)
        return image_base64
    
    def upload_video_to_server(self, video_path: str) -> str:
        """将视频上传到服务器并返回URL"""
        if not video_path or not os.path.exists(video_path):
            raise Exception(f"视频文件不存在: {video_path}")
        
        # 检查文件类型
        file_ext = os.path.splitext(video_path)[1].lower()
        allowed_video_exts = ['.mp4', '.avi', '.mov']
        
        if file_ext not in allowed_video_exts:
            raise Exception(f"不支持的视频格式: {file_ext}，支持的格式: {allowed_video_exts}")
        
        try:
            # 上传到ComfyUI文件服务器
            server_url = "https://ai.comfly.chat"
            upload_url = f"{server_url}/v1/files"
            
            file_name = os.path.basename(video_path)
            
            with open(video_path, 'rb') as file:
                files = {
                    'file': (file_name, file, mimetypes.guess_type(video_path)[0] or 'video/mp4')
                }
                
                # 发送上传请求
                response = requests.post(upload_url, files=files, timeout=300)
                
                if response.status_code == 200:
                    result = response.json()
                    file_url = result.get('url', result.get('id'))
                    if not file_url:
                        raise Exception("服务器未返回文件URL")
                    
                    print(f"视频上传成功: {file_url}")
                    return file_url
                else:
                    raise Exception(f"上传失败: {response.status_code} - {response.text}")
                    
        except Exception as e:
            print(f"上传视频到服务器失败: {e}")
            print("使用本地文件路径")
            # 上传失败，使用本地路径
            return f"file://{video_path}"
    
    def generate_animate_move(self, api_key: str, image: torch.Tensor, video, 
                            mode: str, check_image: bool, seed: int) -> Tuple[str, str]:
        """生成图生动作视频"""
        # 设置API密钥
        self.set_api_key(api_key)
        
        # 处理视频输入 - 支持VIDEO类型和STRING类型
        if isinstance(video, str):
            # STRING类型输入 - 直接使用文件路径
            if not video or not video.strip():
                raise Exception("请输入视频文件路径")
            video_path = video.strip()
        else:
            # VIDEO类型输入 - 从ComfyUI的LoadVideo节点获取路径
            if hasattr(video, 'get'):
                # 如果是字典类型，尝试获取路径
                video_path = video.get('path', '')
            elif hasattr(video, 'filename'):
                # 如果有filename属性
                video_path = video.filename
            elif isinstance(video, tuple) and len(video) > 0:
                # 如果是元组，取第一个元素
                video_path = str(video[0])
            else:
                # 尝试转换为字符串
                video_path = str(video)
            
            if not video_path or not os.path.exists(video_path):
                raise Exception(f"无法获取有效的视频文件路径: {video}")
        
        print("开始处理输入文件...")
        print(f"视频文件路径: {video_path}")
        
        # 上传图像到服务器
        print("正在上传图像...")
        image_url = self.upload_image_to_server(image)
        print(f"图像处理完成")
        
        # 上传视频到服务器
        print("正在上传视频...")
        video_url = self.upload_video_to_server(video_path)
        print(f"视频处理完成: {video_url}")
        
        # 处理随机种子
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        # 构建请求载荷
        payload = {
            "model": "wan2.2-animate-move",
            "input": {
                "image_url": image_url,
                "video_url": video_url
            },
            "parameters": {
                "check_image": check_image,
                "mode": mode,
                "seed": seed
            }
        }
        
        print(f"开始生成图生动作视频")
        print(f"图像URL: {image_url[:100]}...")
        print(f"视频URL: {video_url}")
        print(f"服务模式: {mode} - {self.MODE_DESCRIPTIONS[mode]}")
        print(f"图像检测: {'开启' if check_image else '关闭'}")
        print(f"种子: {seed}")
        
        # 创建任务
        task_id = self.create_task(payload)
        print(f"任务ID: {task_id}")
        
        # 等待任务完成
        result = self.wait_for_completion_with_usage(task_id, model="wan2.2-animate-move")
        
        if isinstance(result, dict):
            video_url = result.get('video_url')
            usage_info = result.get('usage_info', '')
        else:
            # 兼容旧版本返回格式
            video_url = result
            usage_info = "使用信息获取失败"
        
        print(f"视频生成完成: {video_url}")
        print(f"使用信息: {usage_info}")
        
        # 下载视频
        video_path = self.download_video(video_url)
        print(f"视频已保存到: {video_path}")
        
        return (video_path, usage_info)
    
    def wait_for_completion_with_usage(self, task_id: str, timeout: int = 300, model: str = None) -> dict:
        """等待任务完成并返回详细结果信息"""
        # 根据模型类型设置不同的超时时间
        if model and "wan2.2-animate-move" in model:
            timeout = 300  # 万相2.2图生动作模型，设置为5分钟
            print(f"检测到万相2.2图生动作模型，超时时间设置为{timeout}秒")
        
        start_time = time.time()
        poll_interval = 15  # 根据官方文档建议，轮询间隔设置为15秒
        
        print(f"开始等待任务完成，超时时间: {timeout}秒，轮询间隔: {poll_interval}秒")
        
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
                output = result.get('output', {})
                status = output.get('task_status')
                elapsed_time = int(time.time() - start_time)
                
                # 检查是否有video_url字段，即使状态还是RUNNING
                video_url = output.get('results', {}).get('video_url')
                
                if status == 'SUCCEEDED' or video_url:
                    if video_url and status != 'SUCCEEDED':
                        print(f"检测到video_url，任务实际已完成但状态未更新")
                    print_progress_bar(100, 100, prefix='任务完成', suffix=f'耗时:{elapsed_time}s')
                    print(f"任务成功完成，耗时: {elapsed_time}秒")
                    
                    # 提取使用信息
                    usage = result.get('usage', {})
                    usage_info = f"视频时长: {usage.get('video_duration', 'N/A')}秒, 服务模式: {usage.get('video_ratio', 'N/A')}"
                    
                    return {
                        'video_url': video_url,
                        'usage_info': usage_info,
                        'task_id': task_id,
                        'status': status,
                        'elapsed_time': elapsed_time
                    }
                elif status == 'FAILED':
                    error_msg = output.get('message', '未知错误')
                    raise Exception(f"任务失败: {error_msg}")
                elif status == 'CANCELED':
                    raise Exception("任务已取消")
                elif status == 'UNKNOWN':
                    raise Exception("任务不存在或状态未知")
                else:
                    # 显示进度
                    progress = min(int((elapsed_time / timeout) * 100), 95)
                    print_progress_bar(progress, 100, prefix=f'任务状态: {status}', suffix=f'耗时:{elapsed_time}s')
                    
                    if status == 'PENDING':
                        print(f"任务排队中... 已等待 {elapsed_time} 秒")
                    elif status == 'RUNNING':
                        print(f"任务处理中... 已等待 {elapsed_time} 秒")
                        
            else:
                print(f"查询任务状态失败: {response.status_code} - {response.text}")
            
            time.sleep(poll_interval)
        
        # 超时
        raise Exception(f"任务超时，已等待 {timeout} 秒")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AliyunAPIKey": AliyunAPIKey,
    "AliyunTextToVideo": AliyunTextToVideo,
    "AliyunImageToVideo": AliyunImageToVideo,
    "AliyunFirstLastFrameToVideo": AliyunFirstLastFrameToVideo,
    "AliyunVideoEffects": AliyunVideoEffects,
    "FileUploadNode": FileUploadNode,
    "AliyunImageToAnimateMove": AliyunImageToAnimateMove,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AliyunAPIKey": "阿里云API密钥",
    "AliyunTextToVideo": "阿里云文生视频",
    "AliyunImageToVideo": "阿里云图生视频", 
    "AliyunFirstLastFrameToVideo": "阿里云首尾帧生视频",
    "AliyunVideoEffects": "阿里云视频特效",
    "FileUploadNode": "文件上传节点",
    "AliyunImageToAnimateMove": "阿里云万相2.2图生动作",
}