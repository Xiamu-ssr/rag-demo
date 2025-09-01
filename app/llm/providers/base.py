"""模型提供商基类

定义统一的模型提供商接口和抽象方法
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np

from ..config import ModelConfig, ModelType

logger = logging.getLogger(__name__)


class BaseModelProvider(ABC):
    """模型提供商基类"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self._is_loaded = False
        
    @property
    def provider_name(self) -> str:
        """提供商名称"""
        return self.config.provider.value
        
    @property
    def model_id(self) -> str:
        """模型ID"""
        return self.config.model_id
        
    @property
    def model_type(self) -> ModelType:
        """模型类型"""
        return self.config.model_type
        
    @property
    def is_loaded(self) -> bool:
        """模型是否已加载"""
        return self._is_loaded
        
    @abstractmethod
    def load_model(self) -> None:
        """加载模型
        
        Raises:
            RuntimeError: 模型加载失败
        """
        pass
        
    @abstractmethod
    def unload_model(self) -> None:
        """卸载模型"""
        pass
        
    @abstractmethod
    def is_available(self) -> bool:
        """检查模型是否可用
        
        Returns:
            bool: 模型是否可用
        """
        pass
        
    def ensure_loaded(self) -> None:
        """确保模型已加载"""
        if not self._is_loaded:
            logger.info(f"正在加载模型: {self.model_id}")
            self.load_model()
            self._is_loaded = True
            logger.info(f"模型加载完成: {self.model_id}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        return {
            "model_id": self.model_id,
            "name": self.config.name,
            "provider": self.provider_name,
            "type": self.model_type.value,
            "is_loaded": self.is_loaded,
            "is_available": self.is_available(),
            "config": self.config.config
        }


class BaseEmbeddingProvider(BaseModelProvider):
    """嵌入模型提供商基类"""
    
    def __init__(self, config: ModelConfig):
        if config.model_type != ModelType.EMBEDDING:
            raise ValueError(f"配置的模型类型必须是EMBEDDING，当前为: {config.model_type}")
        super().__init__(config)
        self._dimension = None
        
    @property
    def dimension(self) -> Optional[int]:
        """嵌入维度"""
        return self._dimension
        
    @abstractmethod
    def embed_texts(self, texts: List[str], **kwargs) -> np.ndarray:
        """嵌入文本列表
        
        Args:
            texts: 待嵌入的文本列表
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 嵌入向量数组，形状为 [N, dim]
            
        Raises:
            RuntimeError: 嵌入失败
        """
        pass
        
    @abstractmethod
    def embed_query(self, query: str, **kwargs) -> np.ndarray:
        """嵌入查询文本
        
        Args:
            query: 查询文本
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 嵌入向量，形状为 [1, dim]
            
        Raises:
            RuntimeError: 嵌入失败
        """
        pass
        
    def _detect_dimension(self) -> int:
        """检测嵌入维度
        
        Returns:
            int: 嵌入维度
        """
        test_text = ["测试文本用于维度检测"]
        test_embedding = self.embed_texts(test_text)
        dimension = test_embedding.shape[1]
        logger.info(f"检测到模型 {self.model_id} 的嵌入维度: {dimension}")
        return dimension
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息，包含嵌入维度"""
        info = super().get_model_info()
        info["dimension"] = self.dimension
        return info


class BaseChatProvider(BaseModelProvider):
    """聊天模型提供商基类"""
    
    def __init__(self, config: ModelConfig):
        if config.model_type != ModelType.CHAT:
            raise ValueError(f"配置的模型类型必须是CHAT，当前为: {config.model_type}")
        super().__init__(config)
        
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """聊天对话
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            **kwargs: 其他参数
            
        Returns:
            str: 模型回复
            
        Raises:
            RuntimeError: 聊天失败
        """
        pass
        
    @abstractmethod
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs):
        """流式聊天对话
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Yields:
            str: 流式回复片段
            
        Raises:
            RuntimeError: 聊天失败
        """
        pass