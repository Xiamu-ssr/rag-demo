"""LLM模块初始化

提供模型管理和嵌入器功能的统一接口
"""

from .model_manager import (
    ModelManager,
    ModelType,
    get_model_manager
)
from .embedder import (
    BaseEmbedder,
    EmbedderFactory,
    ModernEmbedder,
    get_default_embedder,
    set_default_embedder
)
from .config import (
    ModelConfig,
    DEFAULT_MODEL_CONFIGS
)

__all__ = [
    "ModelManager",
    "ModelType",
    "get_model_manager",
    "BaseEmbedder",
    "EmbedderFactory",
    "ModernEmbedder",
    "get_default_embedder",
    "set_default_embedder",
    "ModelConfig",
    "DEFAULT_MODEL_CONFIGS"
]