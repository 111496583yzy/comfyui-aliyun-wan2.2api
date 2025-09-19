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
    
    def video_to_base64(self, video_tensor: torch.Tensor) -> str:
        """将视频张量转换为base64编码（取第一帧作为图片）"""
        # 视频张量格式通常是 (T, C, H, W) 或 (C, T, H, W)
        if len(video_tensor.shape) == 4:
            # 取第一帧
            first_frame = video_tensor[0] if video_tensor.shape[0] != 3 else video_tensor
        elif len(video_tensor.shape) == 5:
            # (B, T, C, H, W) 格式，取第一帧
            first_frame = video_tensor[0, 0]
        else:
            raise ValueError(f"不支持的视频张量格式: {video_tensor.shape}")
        
        # 确保张量格式为 (C, H, W)
        if len(first_frame.shape) == 3 and first_frame.shape[0] == 3:  # RGB格式
            first_frame = first_frame.permute(1, 2, 0)  # 转换为 (H, W, C)
        
        # 转换为PIL图像
        image_np = (first_frame.cpu().numpy() * 255).astype(np.uint8)
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
        
        # 根据官方文档，检查是否有错误
        if result.get('code'):
            error_code = result.get('code')
            error_msg = result.get('message', '未知错误')
            raise Exception(f"API错误: {error_code} - {error_msg}")
        
        # 检查是否有output字段和task_id
        if 'output' not in result or 'task_id' not in result['output']:
            raise Exception(f"API响应格式错误: {result}")
        
        task_id = result['output']['task_id']
        task_status = result['output'].get('task_status', 'UNKNOWN')
        print(f"任务创建成功，ID: {task_id}, 状态: {task_status}")
        
        return task_id
    
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
                
                # 根据官方文档，video_url在results字段中
                results = result['output'].get('results', {})
                video_url = results.get('video_url')
                
                if status == 'SUCCEEDED' or video_url:
                    if video_url and status != 'SUCCEEDED':
                        print(f"检测到video_url，任务实际已完成但状态未更新")
                    print_progress_bar(100, 100, prefix='任务完成', suffix=f'耗时:{elapsed_time}s')
                    print(f"任务成功完成，耗时: {elapsed_time}秒")
                    
                    # 显示使用量统计信息
                    usage = result.get('usage', {})
                    if usage:
                        video_duration = usage.get('video_duration', 0)
                        video_ratio = usage.get('video_ratio', 'unknown')
                        print(f"生成视频时长: {video_duration}秒, 服务模式: {video_ratio}")
                    
                    return video_url
                elif status == 'FAILED':
                    # 根据官方文档，错误信息可能在output或根级别
                    error_msg = result.get('message') or result['output'].get('message', '未知错误')
                    error_code = result.get('code', '未知错误码')
                    print(f"任务失败: {error_code} - {error_msg}")
                    raise Exception(f"视频生成失败: {error_code} - {error_msg}")
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


class AliyunAnimateMove(AliyunVideoBase):
    """阿里云图生动作节点"""
    
    def __init__(self):
        super().__init__()
        # 图生动作使用相同的API端点
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/video-synthesis"
    
    def image_to_base64_from_url(self, image_url: str) -> str:
        """从URL获取图像并转换为base64编码"""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            
            # 转换为base64
            image_base64 = base64.b64encode(response.content).decode('utf-8')
            return f"data:image/jpeg;base64,{image_base64}"
        except Exception as e:
            raise Exception(f"无法从URL获取图像: {str(e)}")
    
    def video_to_base64_from_url(self, video_url: str) -> str:
        """从URL获取视频并转换为base64编码"""
        try:
            response = requests.get(video_url)
            response.raise_for_status()
            
            # 转换为base64
            video_base64 = base64.b64encode(response.content).decode('utf-8')
            return f"data:video/mp4;base64,{video_base64}"
        except Exception as e:
            raise Exception(f"无法从URL获取视频: {str(e)}")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "forceInput": True,
                    "placeholder": "请输入阿里云API Key"
                }),
                "image": ("IMAGE", {
                    "tooltip": "输入图像，格式：JPG、JPEG、PNG、BMP、WEBP，尺寸：200-4096像素，宽高比：1:3至3:1，大小：不超过5MB"
                }),
                "video_url": ("STRING", {
                    "multiline": False,
                    "default": "https://example.com/reference_video.mp4",
                    "placeholder": "请输入参考视频的URL地址（MP4/AVI/MOV格式，2-30秒，不超过200MB）"
                }),
                "mode": (["wan-std", "wan-pro"], {
                    "default": "wan-std",
                    "tooltip": "wan-std: 标准模式，性价比高，生成速度快\nwan-pro: 专业模式，效果更佳，动画流畅度高"
                }),
                "check_image": ("BOOLEAN", {
                    "default": True,
                    "label_on": "开启图像检测",
                    "label_off": "关闭图像检测",
                    "tooltip": "是否对传入的图片进行检测"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate_video"
    CATEGORY = "Aliyun Video"
    
    def generate_video(self, api_key: str, image: torch.Tensor, video_url: str, 
                      mode: str = "wan-std", check_image: bool = True) -> Tuple[str]:
        """生成图生动作视频"""
        # 设置API密钥
        self.set_api_key(api_key)
        
        # 智能处理输入：判断是图片还是视频
        try:
            # 尝试作为图片处理
            image_base64 = self.image_to_base64(image)
            print("检测到图片输入，使用图片生成动作")
        except Exception as e:
            # 如果图片处理失败，尝试作为视频处理（取第一帧）
            try:
                image_base64 = self.video_to_base64(image)
                print("检测到视频输入，提取第一帧作为图片生成动作")
            except Exception as e2:
                raise Exception(f"输入格式不支持，既不是有效的图片也不是有效的视频: {str(e2)}")
        
        # 验证视频URL格式
        if not video_url.startswith(('http://', 'https://')):
            raise ValueError("视频URL必须以http://或https://开头")
        
        # 验证视频URL不能包含中文字符
        try:
            video_url.encode('ascii')
        except UnicodeEncodeError:
            raise ValueError("视频URL不能包含中文字符，请使用英文URL")
        
        # 验证图像尺寸（ComfyUI的IMAGE类型已经处理了大部分验证）
        if len(image.shape) != 4 or image.shape[0] != 1:
            raise ValueError("图像输入格式错误，请确保输入的是有效的图像")
        
        # 获取图像尺寸进行验证
        _, height, width, _ = image.shape
        if width < 200 or width > 4096 or height < 200 or height > 4096:
            raise ValueError(f"图像尺寸不符合要求，当前尺寸：{width}x{height}，要求：200-4096像素")
        
        aspect_ratio = max(width/height, height/width)
        if aspect_ratio > 3:
            raise ValueError(f"图像宽高比不符合要求，当前比例：{width}:{height}，要求：1:3至3:1")
        
        # 构建payload - 根据官方文档格式
        payload = {
            "model": "wan2.2-animate-move",
            "input": {
                "image_url": image_base64,  # 官方文档要求使用image_url字段
                "video_url": video_url
            },
            "parameters": {
                "check_image": check_image,
                "mode": mode
            }
        }
        
        print(f"开始生成图生动作视频")
        print(f"使用模式: {mode} ({'标准' if mode == 'wan-std' else '专业'})")
        print(f"图像检测: {'开启' if check_image else '关闭'}")
        print(f"参考视频URL: {video_url}")
        print(f"图像尺寸: {width}x{height}")
        print(f"图像宽高比: {width}:{height} ({aspect_ratio:.2f})")
        
        # 显示计费信息
        mode_info = {
            "wan-std": {"name": "标准模式", "price": "0.4元/秒", "description": "生成速度快，性价比高"},
            "wan-pro": {"name": "专业模式", "price": "0.6元/秒", "description": "动画流畅度高，效果更佳"}
        }
        info = mode_info.get(mode, mode_info["wan-std"])
        print(f"计费模式: {info['name']} ({info['price']}) - {info['description']}")
        
        task_id = self.create_task(payload)
        print(f"任务ID: {task_id}")
        
        video_url_result = self.wait_for_completion(task_id, model="wan2.2-animate-move")
        print(f"视频生成完成: {video_url_result}")
        
        video_path = self.download_video(video_url_result)
        print(f"视频已保存到: {video_path}")
        
        return (video_path,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AliyunAPIKey": AliyunAPIKey,
    "AliyunTextToVideo": AliyunTextToVideo,
    "AliyunImageToVideo": AliyunImageToVideo,
    "AliyunFirstLastFrameToVideo": AliyunFirstLastFrameToVideo,
    "AliyunVideoEffects": AliyunVideoEffects,
    "AliyunAnimateMove": AliyunAnimateMove,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AliyunAPIKey": "阿里云API密钥",
    "AliyunTextToVideo": "阿里云文生视频",
    "AliyunImageToVideo": "阿里云图生视频", 
    "AliyunFirstLastFrameToVideo": "阿里云首尾帧生视频",
    "AliyunVideoEffects": "阿里云视频特效",
    "AliyunAnimateMove": "阿里云图生动作",
}