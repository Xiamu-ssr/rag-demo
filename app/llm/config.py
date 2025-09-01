"""模型管理配置模块

定义模型配置、提供商信息和默认设置
"""

from enum import Enum
from typing import Dict, Any, List
from dataclasses import dataclass


class ModelType(Enum):
    """模型类型枚举"""
    EMBEDDING = "embedding"
    CHAT = "chat"
    COMPLETION = "completion"


class ModelProvider(Enum):
    """模型提供商枚举"""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """模型配置数据类"""
    model_id: str
    name: str
    provider: ModelProvider
    model_type: ModelType
    config: Dict[str, Any]
    is_active: bool = True
    is_available: bool = True


# 预定义的模型配置
DEFAULT_MODEL_CONFIGS = {
    # BGE系列嵌入模型
    "bge-m3": ModelConfig(
        model_id="bge-m3",
        name="BGE-M3 多语言嵌入模型",
        provider=ModelProvider.HUGGINGFACE,
        model_type=ModelType.EMBEDDING,
        config={
            "model_name": "BAAI/bge-m3",
            "max_length": 8192,
            "normalize_embeddings": True,
            "batch_size": 64,
            "device": "mps"
        }
    ),
    "bge-large-zh": ModelConfig(
        model_id="bge-large-zh",
        name="BGE-Large 中文嵌入模型",
        provider=ModelProvider.HUGGINGFACE,
        model_type=ModelType.EMBEDDING,
        config={
            "model_name": "BAAI/bge-large-zh-v1.5",
            "max_length": 512,
            "normalize_embeddings": True,
            "batch_size": 32,
            "device": "mps"
        }
    ),
    "bge-base-zh": ModelConfig(
        model_id="bge-base-zh",
        name="BGE-Base 中文嵌入模型",
        provider=ModelProvider.HUGGINGFACE,
        model_type=ModelType.EMBEDDING,
        config={
            "model_name": "BAAI/bge-base-zh-v1.5",
            "max_length": 512,
            "normalize_embeddings": True,
            "batch_size": 64,
            "device": "mps"
        }
    ),
    # OpenAI嵌入模型
    "text-embedding-3-large": ModelConfig(
        model_id="text-embedding-3-large",
        name="OpenAI Text Embedding 3 Large",
        provider=ModelProvider.OPENAI,
        model_type=ModelType.EMBEDDING,
        config={
            "model_name": "text-embedding-3-large",
            "dimensions": 3072,
            "batch_size": 100,
            "encoding_format": "float"
        }
    ),
    "text-embedding-3-small": ModelConfig(
        model_id="text-embedding-3-small",
        name="OpenAI Text Embedding 3 Small",
        provider=ModelProvider.OPENAI,
        model_type=ModelType.EMBEDDING,
        config={
            "model_name": "text-embedding-3-small",
            "dimensions": 1536,
            "batch_size": 100,
            "encoding_format": "float"
        }
    )
}


# 默认设置
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30.0

# 模型加载配置
MODEL_CACHE_DIR = "./models"
MODEL_LOAD_TIMEOUT = 300  # 5分钟

# 性能配置
MAX_CONCURRENT_REQUESTS = 10
MEMORY_LIMIT_GB = 8