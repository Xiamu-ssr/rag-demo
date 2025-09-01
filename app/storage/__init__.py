"""向量存储抽象层

提供统一的向量存储接口，支持多种向量数据库后端
"""

from .base import VectorStore, VectorStoreFactory, VectorRecord, VectorSearchResult

# 导入具体实现以触发注册
try:
    from . import faiss_store
except ImportError:
    pass
    
try:
    from . import milvus_store
except ImportError:
    pass
    
try:
    from . import qdrant_store
except ImportError:
    pass

__all__ = [
    'VectorStore',
    'VectorStoreFactory', 
    'VectorRecord',
    'VectorSearchResult'
]