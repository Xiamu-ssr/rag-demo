"""数据库模型定义"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.sqlite import JSON

from app.core.database import Base


def generate_uuid() -> str:
    """生成UUID字符串"""
    return str(uuid.uuid4())


class Collection(Base):
    """知识库表"""
    __tablename__ = "collections"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text)
    splitting_config = Column(JSON)  # 分段配置
    index_config = Column(JSON)      # 索引配置
    retrieval_config = Column(JSON)  # 检索配置
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    documents = relationship("Document", back_populates="collection", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="collection", cascade="all, delete-orphan")
    index_versions = relationship("IndexVersion", back_populates="collection", cascade="all, delete-orphan")


class Document(Base):
    """文档表"""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    title = Column(String)
    uri = Column(String)  # 文档来源URI
    meta_json = Column(JSON)  # 文档元数据
    status = Column(String, nullable=False, default="ingesting")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 状态约束
    __table_args__ = (
        CheckConstraint(
            "status IN ('ingesting', 'indexed', 'failed', 'deleted')",
            name="check_document_status"
        ),
    )
    
    # 关系
    collection = relationship("Collection", back_populates="documents")
    parents = relationship("Parent", back_populates="document", cascade="all, delete-orphan")


class Parent(Base):
    """父块表"""
    __tablename__ = "parents"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    doc_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    order_no = Column(Integer)  # 在文档中的顺序
    text = Column(Text, nullable=False)
    token_count = Column(Integer)
    headers = Column(Text)  # 标题层级信息
    
    # 关系
    document = relationship("Document", back_populates="parents")
    chunks = relationship("Chunk", back_populates="parent", cascade="all, delete-orphan")


class Chunk(Base):
    """子块表"""
    __tablename__ = "chunks"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    parent_id = Column(String, ForeignKey("parents.id", ondelete="CASCADE"), nullable=False)
    order_no = Column(Integer)  # 在父块中的顺序
    text = Column(Text, nullable=False)
    token_count = Column(Integer)
    
    # 关系
    parent = relationship("Parent", back_populates="chunks")
    embedding = relationship("Embedding", back_populates="chunk", uselist=False, cascade="all, delete-orphan")


class Embedding(Base):
    """向量映射表"""
    __tablename__ = "embeddings"
    
    vector_id = Column(Integer, primary_key=True, autoincrement=True)  # FAISS外部ID
    chunk_id = Column(String, ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, unique=True)
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    model = Column(String, nullable=False)  # 嵌入模型名称
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    chunk = relationship("Chunk", back_populates="embedding")
    collection = relationship("Collection", back_populates="embeddings")


class IndexVersion(Base):
    """索引版本表"""
    __tablename__ = "index_versions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    collection_id = Column(String, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False)
    status = Column(String, nullable=False, default="building")
    faiss_path = Column(String, nullable=False)  # FAISS索引文件路径
    meta_json = Column(JSON)  # 索引元信息
    created_at = Column(DateTime, default=datetime.utcnow)
    activated_at = Column(DateTime)  # 激活时间
    
    # 状态约束
    __table_args__ = (
        CheckConstraint(
            "status IN ('building', 'active', 'failed')",
            name="check_index_status"
        ),
    )
    
    # 关系
    collection = relationship("Collection", back_populates="index_versions")


class ModelConfig(Base):
    """模型配置表"""
    __tablename__ = "model_configs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    model_id = Column(String, nullable=False, unique=True)  # 模型唯一标识
    name = Column(String, nullable=False)  # 模型显示名称
    provider = Column(String, nullable=False)  # 模型提供商
    model_type = Column(String, nullable=False)  # 模型类型
    config = Column(JSON)  # 模型配置参数
    is_active = Column(String, nullable=False, default="true")  # 是否激活
    is_default = Column(String, nullable=False, default="false")  # 是否为默认模型
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 约束
    __table_args__ = (
        CheckConstraint(
            "provider IN ('huggingface', 'openai', 'anthropic', 'google')",
            name="check_model_provider"
        ),
        CheckConstraint(
            "model_type IN ('embedding', 'chat', 'completion')",
            name="check_model_type"
        ),
        CheckConstraint(
            "is_active IN ('true', 'false')",
            name="check_model_is_active"
        ),
        CheckConstraint(
            "is_default IN ('true', 'false')",
            name="check_model_is_default"
        ),
    )