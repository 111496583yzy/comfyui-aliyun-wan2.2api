#!/usr/bin/env python3
"""
测试文件上传功能
"""

import requests
import os
import tempfile
from PIL import Image
import numpy as np

def upload_file_to_server(file_path: str, upload_url: str = "https://ai.kefan.cn/api/upload/local") -> str:
    """
    上传文件到服务器并获取在线URL
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 检查文件大小（最大200MB）
        file_size = os.path.getsize(file_path)
        if file_size > 200 * 1024 * 1024:  # 200MB
            raise ValueError(f"文件过大: {file_size / (1024*1024):.1f}MB，最大支持200MB")
        
        print(f"开始上传文件: {os.path.basename(file_path)} ({file_size / (1024*1024):.1f}MB)")
        
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            response = requests.post(upload_url, files=files, timeout=300)  # 5分钟超时
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('code') == 200:
                online_url = result.get('data')
                print(f"文件上传成功: {online_url}")
                return online_url
            else:
                raise Exception(f"上传失败: {result.get('message', '未知错误')}")
        else:
            raise Exception(f"上传请求失败: HTTP {response.status_code}")
            
    except requests.exceptions.Timeout:
        raise Exception("上传超时，请检查网络连接或文件大小")
    except requests.exceptions.RequestException as e:
        raise Exception(f"网络请求失败: {str(e)}")
    except Exception as e:
        raise Exception(f"上传文件失败: {str(e)}")

def create_test_image():
    """创建一个测试图像"""
    # 创建一个简单的测试图像
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    
    # 保存到临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_path = temp_file.name
    temp_file.close()
    
    img.save(temp_path, format='PNG')
    print(f"创建测试图像: {temp_path}")
    
    return temp_path

def main():
    """主函数"""
    print("测试文件上传功能...")
    
    try:
        # 创建测试图像
        test_image_path = create_test_image()
        
        # 上传测试图像
        online_url = upload_file_to_server(test_image_path)
        
        print(f"测试成功！在线URL: {online_url}")
        
        # 清理临时文件
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)
            print("已清理临时文件")
            
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    main()
