"""检索模块初始化

提供向量检索器、MMR去冗余算法和多库融合算法的统一接口
"""

from .vector_retriever import (
    RetrievalResult,
    RetrievalConfig,
    VectorRetriever,
    get_vector_retriever
)

from .mmr import (
    MMRConfig,
    MMRSelector,
    get_mmr_selector,
    set_mmr_selector
)

from .fusion import (
    FusionMethod,
    FusionConfig,
    SourceResults,
    ResultFusion,
    get_result_fusion,
    set_result_fusion
)

__all__ = [
    # 向量检索器
    "RetrievalResult",
    "RetrievalConfig",
    "VectorRetriever",
    "get_vector_retriever",
    
    # MMR去冗余
    "MMRConfig",
    "MMRSelector",
    "get_mmr_selector",
    "set_mmr_selector",
    
    # 结果融合
    "FusionMethod",
    "FusionConfig",
    "SourceResults",
    "ResultFusion",
    "get_result_fusion",
    "set_result_fusion"
]