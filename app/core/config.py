"""应用配置模块"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基础配置
    app_name: str = "RAG Demo"
    debug: bool = Field(default=False, env="DEBUG")
    
    # 数据库配置
    database_url: str = Field(default="sqlite:///./rag_demo.db", env="DATABASE_URL")
    
    # 向量索引配置
    index_base_path: str = Field(default="./data/indices", env="INDEX_BASE_PATH")
    
    # AI模型配置
    embedding_model: str = Field(default="bge-m3", env="EMBEDDING_MODEL")
    embedding_api_url: Optional[str] = Field(default=None, env="EMBEDDING_API_URL")
    embedding_api_key: Optional[str] = Field(default=None, env="EMBEDDING_API_KEY")
    
    # 文档处理配置
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    supported_file_types: list[str] = ["pdf", "txt", "html", "md", "docx"]
    
    # 检索配置
    default_top_k: int = Field(default=3, env="DEFAULT_TOP_K")
    default_min_score: float = Field(default=0.5, env="DEFAULT_MIN_SCORE")
    
    # 分段配置
    default_chunk_size: int = Field(default=512, env="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=80, env="DEFAULT_CHUNK_OVERLAP")
    default_parent_max_tokens: int = Field(default=2000, env="DEFAULT_PARENT_MAX_TOKENS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# 全局配置实例
settings = Settings()

# 确保索引目录存在
os.makedirs(settings.index_base_path, exist_ok=True)