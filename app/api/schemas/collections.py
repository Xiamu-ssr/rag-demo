"""知识库相关的数据模式"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class CollectionBase(BaseModel):
    """知识库基础模式"""
    name: str = Field(..., description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")


class CollectionCreate(CollectionBase):
    """创建知识库请求模式"""
    splitting_config: Optional[Dict[str, Any]] = Field(
        default={
            "mode": "parent_child",
            "parent": {"type": "paragraph", "max_tokens": 2000},
            "child": {"chunk_size": 512, "chunk_overlap": 80}
        },
        description="分段配置"
    )
    index_config: Optional[Dict[str, Any]] = Field(
        default={"embedding_model": "bge-m3"},
        description="索引配置"
    )
    retrieval_config: Optional[Dict[str, Any]] = Field(
        default={
            "mode": "vector",
            "vector": {"top_k": 3, "min_score": 0.5}
        },
        description="检索配置"
    )


class CollectionUpdate(BaseModel):
    """更新知识库请求模式"""
    name: Optional[str] = Field(None, description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")
    retrieval_config: Optional[Dict[str, Any]] = Field(None, description="检索配置")


class CollectionResponse(CollectionBase):
    """知识库响应模式"""
    id: str = Field(..., description="知识库ID")
    splitting_config: Optional[Dict[str, Any]] = Field(None, description="分段配置")
    index_config: Optional[Dict[str, Any]] = Field(None, description="索引配置")
    retrieval_config: Optional[Dict[str, Any]] = Field(None, description="检索配置")
    document_count: int = Field(0, description="文档数量")
    chunk_count: int = Field(0, description="分段数量")
    created_at: datetime = Field(..., description="创建时间")
    
    class Config:
        from_attributes = True


class CollectionListResponse(BaseModel):
    """知识库列表响应模式"""
    collections: List[CollectionResponse] = Field(..., description="知识库列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页码")
    size: int = Field(..., description="每页大小")