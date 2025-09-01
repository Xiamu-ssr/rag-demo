"""嵌入器模块

提供文本嵌入功能，支持多种嵌入模型的接口适配。
现已重构为使用模型管理中心进行统一管理。
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Protocol, Dict, Any, Union

import numpy as np

from .model_manager import get_model_manager
from .config import ModelType

logger = logging.getLogger(__name__)


class EmbedderProtocol(Protocol):
    """嵌入器协议接口"""
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        ...
    
    def embed_query(self, query: str) -> List[float]:
        """嵌入单个查询"""
        ...
    
    def get_dimension(self) -> int:
        """获取嵌入维度"""
        ...


class BaseEmbedder(ABC):
    """嵌入器基类
    
    现已重构为使用模型管理中心，提供统一的嵌入功能接口。
    """
    
    def __init__(self, model_id: str = None, batch_size: int = 32, max_retries: int = 3, timeout: int = 300):
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self._model_manager = get_model_manager()
        self._dimension = None
        
    def _get_embedding_provider(self):
        """获取嵌入模型提供商"""
        if self.model_id:
            provider = self._model_manager.get_provider(self.model_id)
            if provider is None:
                # 尝试加载模型
                self._model_manager.load_model(self.model_id)
                provider = self._model_manager.get_provider(self.model_id)
            return provider
        else:
            # 使用当前嵌入模型
            return self._model_manager.get_current_embedding_model()
    
    def get_dimension(self) -> int:
        """获取嵌入维度"""
        if self._dimension is None:
            provider = self._get_embedding_provider()
            if provider:
                # 通过测试嵌入获取维度
                test_embedding = provider.embed_query("test")
                self._dimension = len(test_embedding)
                logger.info(f"Model dimension: {self._dimension}")
            else:
                raise RuntimeError("No embedding provider available")
        return self._dimension
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本
        
        使用模型管理中心的嵌入提供商进行批量嵌入
        """
        if not texts:
            return []
        
        provider = self._get_embedding_provider()
        if not provider:
            raise RuntimeError("No embedding provider available")
        
        all_embeddings = []
        current_batch_size = self.batch_size
        
        for i in range(0, len(texts), current_batch_size):
            batch = texts[i:i + current_batch_size]
            
            # 重试机制
            for attempt in range(self.max_retries):
                try:
                    start_time = time.time()
                    
                    # 使用提供商的批量嵌入方法
                    batch_embeddings = provider.embed_texts(batch)
                    
                    process_time = time.time() - start_time
                    logger.debug(f"Batch {i//current_batch_size + 1} processed in {process_time:.2f}s")
                    
                    all_embeddings.extend(batch_embeddings)
                    break
                    
                except Exception as e:
                    logger.warning(f"Batch embedding failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to embed batch after {self.max_retries} attempts")
                        raise
                    
                    # 批大小回退策略
                    if current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        logger.info(f"Reducing batch size to {current_batch_size}")
                    
                    time.sleep(2 ** attempt)  # 指数退避
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """嵌入单个查询"""
        provider = self._get_embedding_provider()
        if not provider:
            raise RuntimeError("No embedding provider available")
        
        return provider.embed_query(query)
        
    def embed(self, texts: List[str], *, batch_size: int = 64) -> np.ndarray:
        """批量嵌入文本（兼容旧接口）
        
        Args:
            texts: 待嵌入的文本列表
            batch_size: 批处理大小
            
        Returns:
            形状为 [N, dim] 的向量数组
        """
        embeddings = self.embed_texts(texts)
        return np.array(embeddings) if embeddings else np.array([])
        
class ModernEmbedder(BaseEmbedder):
    """现代化嵌入器实现
    
    使用模型管理中心进行统一管理的嵌入器。
    """
    pass


class EmbedderFactory:
    """嵌入器工厂类
    
    负责创建和管理不同类型的嵌入器实例。
    现已重构为使用模型管理中心。
    """
    
    @classmethod
    def create_embedder(cls, model_id: str = None, **kwargs) -> BaseEmbedder:
        """创建嵌入器实例
        
        Args:
            model_id: 模型ID，如果为None则使用默认模型
            **kwargs: 其他参数
            
        Returns:
            嵌入器实例
        """
        return ModernEmbedder(model_id=model_id, **kwargs)
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """获取支持的模型列表"""
        model_manager = get_model_manager()
        return model_manager.list_models(model_type=ModelType.EMBEDDING)


# 默认嵌入器实例（延迟初始化）
_default_embedder: Optional[BaseEmbedder] = None


def get_default_embedder() -> BaseEmbedder:
    """获取默认嵌入器实例"""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = EmbedderFactory.create_embedder("bge-m3")
    return _default_embedder


def set_default_embedder(embedder: BaseEmbedder):
    """设置默认嵌入器"""
    global _default_embedder
    _default_embedder = embedder