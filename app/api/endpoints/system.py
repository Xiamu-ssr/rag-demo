"""系统配置和状态API端点"""

from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/status")
async def get_system_status():
    """获取系统状态"""
    return {
        "status": "running",
        "version": "1.0.0",
        "embedding_model": settings.embedding_model,
        "collections_count": 0,  # 待实现
        "documents_count": 0,    # 待实现
        "chunks_count": 0,       # 待实现
        "index_size_total": 0,   # 待实现
        "available_models": ["bge-m3"]
    }


@router.get("/config/templates")
async def get_config_templates():
    """获取配置模板"""
    return {
        "splitting": {
            "parent_child": {
                "mode": "parent_child",
                "parent": {"type": "paragraph", "max_tokens": 2000},
                "child": {"chunk_size": 512, "chunk_overlap": 80}
            },
            "flat": {
                "mode": "flat",
                "parent": {"type": "document", "max_tokens": 10000},
                "child": {"chunk_size": 512, "chunk_overlap": 80}
            }
        },
        "index": {
            "default": {
                "embedding_model": "bge-m3",
                "normalize": True
            }
        },
        "retrieval": {
            "vector": {
                "mode": "vector",
                "vector": {"top_k": 3, "min_score": 0.5}
            },
            "hybrid": {
                "mode": "hybrid_weighted",
                "weights": {"vector": 0.7, "lexical": 0.3}
            }
        }
    }