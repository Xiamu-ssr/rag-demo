"""OpenAI模型提供商实现

支持OpenAI的嵌入模型和聊天模型
"""

import logging
import os
from typing import List, Dict, Any, Optional

import numpy as np

from .base import BaseEmbeddingProvider, BaseChatProvider
from ..config import ModelConfig, ModelType

logger = logging.getLogger(__name__)

# 延迟导入OpenAI，避免在没有安装时报错
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI库未安装，OpenAI提供商将不可用")


class OpenAIProvider(BaseEmbeddingProvider):
    """OpenAI嵌入模型提供商"""
    
    def __init__(self, config: ModelConfig):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI库未安装，请运行: pip install openai")
            
        super().__init__(config)
        self.model_name = config.config.get("model_name", config.model_id)
        self.api_key = config.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = config.config.get("base_url") or os.getenv("OPENAI_BASE_URL")
        self.dimensions = config.config.get("dimensions")
        self.encoding_format = config.config.get("encoding_format", "float")
        self.batch_size = config.config.get("batch_size", 100)
        
        if not self.api_key:
            raise ValueError("OpenAI API密钥未设置，请设置OPENAI_API_KEY环境变量或在配置中提供api_key")
            
    def load_model(self) -> None:
        """加载OpenAI客户端"""
        try:
            # 创建OpenAI客户端
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
                
            self.model = OpenAI(**client_kwargs)
            
            # 设置嵌入维度
            if self.dimensions:
                self._dimension = self.dimensions
            else:
                # 检测嵌入维度
                self._dimension = self._detect_dimension()
                
            logger.info(f"OpenAI客户端初始化完成: {self.model_name}")
            
        except Exception as e:
            logger.error(f"初始化OpenAI客户端失败: {e}")
            raise RuntimeError(f"初始化OpenAI客户端失败: {e}")
            
    def unload_model(self) -> None:
        """卸载模型"""
        if self.model is not None:
            self.model = None
            self._is_loaded = False
            self._dimension = None
            logger.info(f"已卸载OpenAI客户端: {self.model_name}")
            
    def is_available(self) -> bool:
        """检查模型是否可用"""
        if not OPENAI_AVAILABLE or not self.api_key:
            return False
            
        try:
            # 尝试创建客户端并进行简单测试
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
                
            test_client = OpenAI(**client_kwargs)
            
            # 尝试嵌入一个简单文本来测试连接
            test_response = test_client.embeddings.create(
                model=self.model_name,
                input=["test"],
                encoding_format=self.encoding_format
            )
            
            return len(test_response.data) > 0
            
        except Exception as e:
            logger.warning(f"OpenAI模型不可用: {self.model_name}, 错误: {e}")
            return False
            
    def embed_texts(self, texts: List[str], **kwargs) -> np.ndarray:
        """嵌入文本列表
        
        Args:
            texts: 待嵌入的文本列表
            **kwargs: 其他参数，支持:
                - batch_size: 批处理大小
                - dimensions: 嵌入维度（仅部分模型支持）
                
        Returns:
            np.ndarray: 嵌入向量数组，形状为 [N, dim]
        """
        if not texts:
            return np.array([])
            
        self.ensure_loaded()
        
        try:
            # 获取参数
            batch_size = kwargs.get("batch_size", self.batch_size)
            dimensions = kwargs.get("dimensions", self.dimensions)
            
            all_embeddings = []
            
            # 分批处理
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 构建请求参数
                request_params = {
                    "model": self.model_name,
                    "input": batch_texts,
                    "encoding_format": self.encoding_format
                }
                
                if dimensions:
                    request_params["dimensions"] = dimensions
                    
                # 调用OpenAI API
                response = self.model.embeddings.create(**request_params)
                
                # 提取嵌入向量
                batch_embeddings = []
                for data in response.data:
                    embedding = np.array(data.embedding, dtype=np.float32)
                    batch_embeddings.append(embedding)
                    
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"OpenAI嵌入批次完成: {len(batch_texts)} 个文本")
                
            # 转换为numpy数组
            embeddings = np.array(all_embeddings, dtype=np.float32)
            
            logger.debug(f"OpenAI嵌入完成: {len(texts)} 个文本, 输出形状: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI嵌入失败: {e}")
            raise RuntimeError(f"OpenAI嵌入失败: {e}")
            
    def embed_query(self, query: str, **kwargs) -> np.ndarray:
        """嵌入查询文本
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 嵌入向量，形状为 [1, dim]
        """
        if not query.strip():
            raise ValueError("查询文本不能为空")
            
        embeddings = self.embed_texts([query], **kwargs)
        return embeddings
        
    def get_model_info(self) -> dict:
        """获取模型详细信息"""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "api_key_set": bool(self.api_key),
            "base_url": self.base_url,
            "encoding_format": self.encoding_format,
            "batch_size": self.batch_size
        })
        if self.dimensions:
            info["dimensions"] = self.dimensions
        return info


class OpenAIChatProvider(BaseChatProvider):
    """OpenAI聊天模型提供商"""
    
    def __init__(self, config: ModelConfig):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI库未安装，请运行: pip install openai")
            
        super().__init__(config)
        self.model_name = config.config.get("model_name", config.model_id)
        self.api_key = config.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = config.config.get("base_url") or os.getenv("OPENAI_BASE_URL")
        self.temperature = config.config.get("temperature", 0.7)
        self.max_tokens = config.config.get("max_tokens")
        
        if not self.api_key:
            raise ValueError("OpenAI API密钥未设置")
            
    def load_model(self) -> None:
        """加载OpenAI客户端"""
        try:
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
                
            self.model = OpenAI(**client_kwargs)
            logger.info(f"OpenAI聊天客户端初始化完成: {self.model_name}")
            
        except Exception as e:
            logger.error(f"初始化OpenAI聊天客户端失败: {e}")
            raise RuntimeError(f"初始化OpenAI聊天客户端失败: {e}")
            
    def unload_model(self) -> None:
        """卸载模型"""
        if self.model is not None:
            self.model = None
            self._is_loaded = False
            logger.info(f"已卸载OpenAI聊天客户端: {self.model_name}")
            
    def is_available(self) -> bool:
        """检查模型是否可用"""
        if not OPENAI_AVAILABLE or not self.api_key:
            return False
            
        try:
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
                
            test_client = OpenAI(**client_kwargs)
            
            # 尝试简单的聊天测试
            test_response = test_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            
            return bool(test_response.choices)
            
        except Exception as e:
            logger.warning(f"OpenAI聊天模型不可用: {self.model_name}, 错误: {e}")
            return False
            
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """聊天对话"""
        self.ensure_loaded()
        
        try:
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature)
            }
            
            if self.max_tokens:
                request_params["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)
                
            response = self.model.chat.completions.create(**request_params)
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI聊天失败: {e}")
            raise RuntimeError(f"OpenAI聊天失败: {e}")
            
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs):
        """流式聊天对话"""
        self.ensure_loaded()
        
        try:
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "stream": True
            }
            
            if self.max_tokens:
                request_params["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)
                
            stream = self.model.chat.completions.create(**request_params)
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI流式聊天失败: {e}")
            raise RuntimeError(f"OpenAI流式聊天失败: {e}")