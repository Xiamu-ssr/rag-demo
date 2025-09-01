"""HuggingFace模型提供商实现

支持HuggingFace Hub上的各种模型，主要是嵌入模型
"""

import logging
import time
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseEmbeddingProvider
from ..config import ModelConfig

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseEmbeddingProvider):
    """HuggingFace嵌入模型提供商"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_name = config.config.get("model_name", config.model_id)
        self.device = config.config.get("device", "auto")
        self.normalize_embeddings = config.config.get("normalize_embeddings", True)
        self.max_length = config.config.get("max_length", 512)
        self.batch_size = config.config.get("batch_size", 64)
        
    def load_model(self) -> None:
        """加载HuggingFace模型"""
        try:
            start_time = time.time()
            
            # 加载SentenceTransformer模型
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.config.config.get("cache_dir")
            )
            
            # 设置为评估模式
            self.model.eval()
            
            # 标记为已加载，避免递归调用
            self._is_loaded = True
            
            # 检测嵌入维度
            self._dimension = self._detect_dimension()
            
            load_time = time.time() - start_time
            logger.info(f"HuggingFace模型加载完成: {self.model_name}, 耗时: {load_time:.2f}s")
            
        except Exception as e:
            self._is_loaded = False
            logger.error(f"加载HuggingFace模型失败: {self.model_name}, 错误: {e}")
            raise RuntimeError(f"加载HuggingFace模型失败: {e}")
            
    def unload_model(self) -> None:
        """卸载模型"""
        if self.model is not None:
            # 清理GPU内存
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
            del self.model
            self.model = None
            self._is_loaded = False
            self._dimension = None
            logger.info(f"已卸载HuggingFace模型: {self.model_name}")
            
    def is_available(self) -> bool:
        """检查模型是否可用"""
        try:
            # 尝试创建模型实例来检查可用性
            test_model = SentenceTransformer(self.model_name)
            del test_model
            return True
        except Exception as e:
            logger.warning(f"HuggingFace模型不可用: {self.model_name}, 错误: {e}")
            return False
            
    def embed_texts(self, texts: List[str], **kwargs) -> np.ndarray:
        """嵌入文本列表
        
        Args:
            texts: 待嵌入的文本列表
            **kwargs: 其他参数，支持:
                - batch_size: 批处理大小
                - normalize_embeddings: 是否归一化
                - show_progress_bar: 是否显示进度条
                
        Returns:
            np.ndarray: 嵌入向量数组，形状为 [N, dim]
        """
        if not texts:
            return np.array([])
            
        self.ensure_loaded()
        
        try:
            # 获取参数
            batch_size = kwargs.get("batch_size", self.batch_size)
            normalize = kwargs.get("normalize_embeddings", self.normalize_embeddings)
            show_progress = kwargs.get("show_progress_bar", False)
            
            # 使用sentence-transformers进行嵌入
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # 确保返回float32类型
            embeddings = embeddings.astype(np.float32)
            
            logger.debug(f"嵌入完成: {len(texts)} 个文本, 输出形状: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"HuggingFace嵌入失败: {e}")
            raise RuntimeError(f"HuggingFace嵌入失败: {e}")
            
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
            
        # 对于查询，通常使用单独的处理
        embeddings = self.embed_texts([query], **kwargs)
        return embeddings
        
    def embed_batch_with_retry(self, texts: List[str], max_retries: int = 3, **kwargs) -> np.ndarray:
        """带重试机制的批量嵌入
        
        Args:
            texts: 待嵌入的文本列表
            max_retries: 最大重试次数
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 嵌入向量数组
        """
        last_exception = None
        current_batch_size = kwargs.get("batch_size", self.batch_size)
        
        for attempt in range(max_retries):
            try:
                return self.embed_texts(texts, batch_size=current_batch_size, **kwargs)
                
            except Exception as e:
                last_exception = e
                logger.warning(f"嵌入失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # 减小批处理大小
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.info(f"减小批处理大小至: {current_batch_size}")
                    
                    # 指数退避
                    wait_time = 2 ** attempt
                    logger.info(f"等待 {wait_time}s 后重试...")
                    time.sleep(wait_time)
                    
        raise RuntimeError(f"嵌入失败，已重试 {max_retries} 次: {last_exception}")
        
    def get_model_info(self) -> dict:
        """获取模型详细信息"""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size
        })
        return info