"""自定义异常类"""

from typing import Any, Dict, Optional


class RAGDemoException(Exception):
    """RAG Demo基础异常类"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class CollectionNotFoundError(RAGDemoException):
    """知识库未找到异常"""
    pass


class DocumentNotFoundError(RAGDemoException):
    """文档未找到异常"""
    pass


class DocumentParsingError(RAGDemoException):
    """文档解析异常"""
    pass


class EmbeddingError(RAGDemoException):
    """向量化异常"""
    pass


class IndexError(RAGDemoException):
    """索引操作异常"""
    pass


class RetrievalError(RAGDemoException):
    """检索异常"""
    pass


class ValidationError(RAGDemoException):
    """数据验证异常"""
    pass


class ConfigurationError(RAGDemoException):
    """配置异常"""
    pass