"""工具模块初始化

提供向量处理工具函数的统一接口
"""

from .vector_utils import (
    NormalizationType,
    SimilarityType,
    VectorProcessor,
    get_vector_processor,
    set_vector_processor,
    normalize_vector,
    cosine_similarity,
    euclidean_distance
)

__all__ = [
    # 向量处理
    "NormalizationType",
    "SimilarityType",
    "VectorProcessor",
    "get_vector_processor",
    "set_vector_processor",
    
    # 便捷函数
    "normalize_vector",
    "cosine_similarity",
    "euclidean_distance"
]