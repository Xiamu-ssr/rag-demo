"""向量存储抽象层基类

定义向量存储的统一接口，支持多种向量数据库后端
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    vector: Optional[np.ndarray] = None


@dataclass
class VectorRecord:
    """向量记录"""
    id: str
    vector: np.ndarray
    metadata: Optional[Dict[str, Any]] = None


class VectorStore(ABC):
    """向量存储抽象基类
    
    定义向量存储的统一接口，支持向量的增删改查和相似性搜索
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化向量存储
        
        Args:
            config: 存储配置
        """
        self.config = config
        self._is_initialized = False
        
    @abstractmethod
    def initialize(self) -> None:
        """初始化存储
        
        创建必要的索引、连接等资源
        """
        pass
        
    @abstractmethod
    def close(self) -> None:
        """关闭存储
        
        释放资源，关闭连接
        """
        pass
        
    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int, 
                         metadata_schema: Optional[Dict[str, str]] = None) -> None:
        """创建集合
        
        Args:
            collection_name: 集合名称
            dimension: 向量维度
            metadata_schema: 元数据模式定义
        """
        pass
        
    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """删除集合
        
        Args:
            collection_name: 集合名称
        """
        pass
        
    @abstractmethod
    def list_collections(self) -> List[str]:
        """列出所有集合
        
        Returns:
            List[str]: 集合名称列表
        """
        pass
        
    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在
        
        Args:
            collection_name: 集合名称
            
        Returns:
            bool: 集合是否存在
        """
        pass
        
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """获取集合信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            Dict[str, Any]: 集合信息
        """
        pass
        
    @abstractmethod
    def insert_vectors(self, collection_name: str, vectors: List[VectorRecord]) -> List[str]:
        """插入向量
        
        Args:
            collection_name: 集合名称
            vectors: 向量记录列表
            
        Returns:
            List[str]: 插入的向量ID列表
        """
        pass
        
    @abstractmethod
    def update_vectors(self, collection_name: str, vectors: List[VectorRecord]) -> None:
        """更新向量
        
        Args:
            collection_name: 集合名称
            vectors: 向量记录列表
        """
        pass
        
    @abstractmethod
    def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> None:
        """删除向量
        
        Args:
            collection_name: 集合名称
            vector_ids: 要删除的向量ID列表
        """
        pass
        
    @abstractmethod
    def get_vector(self, collection_name: str, vector_id: str) -> Optional[VectorRecord]:
        """获取单个向量
        
        Args:
            collection_name: 集合名称
            vector_id: 向量ID
            
        Returns:
            Optional[VectorRecord]: 向量记录，如果不存在则返回None
        """
        pass
        
    @abstractmethod
    def search_vectors(self, collection_name: str, query_vector: np.ndarray, 
                      top_k: int = 10, filter_expr: Optional[str] = None,
                      **kwargs) -> List[VectorSearchResult]:
        """向量相似性搜索
        
        Args:
            collection_name: 集合名称
            query_vector: 查询向量
            top_k: 返回结果数量
            filter_expr: 过滤表达式
            **kwargs: 其他搜索参数
            
        Returns:
            List[VectorSearchResult]: 搜索结果列表
        """
        pass
        
    @abstractmethod
    def batch_search_vectors(self, collection_name: str, query_vectors: List[np.ndarray],
                           top_k: int = 10, filter_expr: Optional[str] = None,
                           **kwargs) -> List[List[VectorSearchResult]]:
        """批量向量相似性搜索
        
        Args:
            collection_name: 集合名称
            query_vectors: 查询向量列表
            top_k: 返回结果数量
            filter_expr: 过滤表达式
            **kwargs: 其他搜索参数
            
        Returns:
            List[List[VectorSearchResult]]: 每个查询向量的搜索结果列表
        """
        pass
        
    @abstractmethod
    def count_vectors(self, collection_name: str, filter_expr: Optional[str] = None) -> int:
        """统计向量数量
        
        Args:
            collection_name: 集合名称
            filter_expr: 过滤表达式
            
        Returns:
            int: 向量数量
        """
        pass
        
    def is_initialized(self) -> bool:
        """检查是否已初始化
        
        Returns:
            bool: 是否已初始化
        """
        return self._is_initialized
        
    def get_storage_type(self) -> str:
        """获取存储类型
        
        Returns:
            str: 存储类型名称
        """
        return self.__class__.__name__.lower().replace('vectorstore', '')
        
    def get_config(self) -> Dict[str, Any]:
        """获取配置
        
        Returns:
            Dict[str, Any]: 存储配置
        """
        return self.config.copy()
        
    def validate_vector_dimension(self, collection_name: str, vector: np.ndarray) -> bool:
        """验证向量维度
        
        Args:
            collection_name: 集合名称
            vector: 向量
            
        Returns:
            bool: 维度是否匹配
        """
        try:
            collection_info = self.get_collection_info(collection_name)
            expected_dim = collection_info.get('dimension')
            if expected_dim and vector.shape[-1] != expected_dim:
                return False
            return True
        except Exception:
            return False
            
    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """归一化向量
        
        Args:
            vector: 输入向量
            
        Returns:
            np.ndarray: 归一化后的向量
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
        
    def batch_normalize_vectors(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
        """批量归一化向量
        
        Args:
            vectors: 向量列表
            
        Returns:
            List[np.ndarray]: 归一化后的向量列表
        """
        return [self.normalize_vector(vector) for vector in vectors]


class VectorStoreFactory:
    """向量存储工厂类"""
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, storage_type: str, storage_class: type) -> None:
        """注册存储类型
        
        Args:
            storage_type: 存储类型名称
            storage_class: 存储类
        """
        cls._registry[storage_type.lower()] = storage_class
        
    @classmethod
    def create(cls, storage_type: str, config: Dict[str, Any]) -> VectorStore:
        """创建向量存储实例
        
        Args:
            storage_type: 存储类型
            config: 配置
            
        Returns:
            VectorStore: 向量存储实例
            
        Raises:
            ValueError: 不支持的存储类型
        """
        storage_type = storage_type.lower()
        if storage_type not in cls._registry:
            raise ValueError(f"不支持的存储类型: {storage_type}")
            
        storage_class = cls._registry[storage_type]
        return storage_class(config)
        
    @classmethod
    def list_supported_types(cls) -> List[str]:
        """列出支持的存储类型
        
        Returns:
            List[str]: 支持的存储类型列表
        """
        return list(cls._registry.keys())
        
    @classmethod
    def is_supported(cls, storage_type: str) -> bool:
        """检查是否支持指定的存储类型
        
        Args:
            storage_type: 存储类型
            
        Returns:
            bool: 是否支持
        """
        return storage_type.lower() in cls._registry