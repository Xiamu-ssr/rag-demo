"""模型管理器

统一管理各种模型的加载、切换和配置
"""

import logging
from typing import Dict, List, Optional, Type, Any, Union
from datetime import datetime

from sqlalchemy.orm import Session

from .config import ModelConfig, ModelProvider, ModelType, DEFAULT_MODEL_CONFIGS
from .providers.base import BaseModelProvider, BaseEmbeddingProvider, BaseChatProvider
from .providers.huggingface_provider import HuggingFaceProvider
from .providers.openai_provider import OpenAIProvider
from .providers.openai_chat_provider import OpenAIChatProvider
from ..core.database import get_session
from ..db.models import ModelConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器
    
    负责模型的注册、加载、切换和生命周期管理
    """
    
    def __init__(self):
        self._providers: Dict[str, BaseModelProvider] = {}
        self._provider_classes: Dict[ModelProvider, Dict[ModelType, Type[BaseModelProvider]]] = {
            ModelProvider.HUGGINGFACE: {
                ModelType.EMBEDDING: HuggingFaceProvider
            },
            ModelProvider.OPENAI: {
                ModelType.EMBEDDING: OpenAIProvider,
                ModelType.CHAT: OpenAIChatProvider
            }
        }
        self._current_embedding_model: Optional[str] = None
        self._current_chat_model: Optional[str] = None
        
    def register_provider_class(self, provider: ModelProvider, model_type: ModelType, 
                               provider_class: Type[BaseModelProvider]) -> None:
        """注册提供商类
        
        Args:
            provider: 提供商类型
            model_type: 模型类型
            provider_class: 提供商实现类
        """
        if provider not in self._provider_classes:
            self._provider_classes[provider] = {}
        self._provider_classes[provider][model_type] = provider_class
        logger.info(f"注册提供商类: {provider.value}/{model_type.value} -> {provider_class.__name__}")
        
    def load_model(self, model_id: str, config: Optional[ModelConfig] = None) -> BaseModelProvider:
        """加载模型
        
        Args:
            model_id: 模型ID
            config: 模型配置，如果为None则从数据库或默认配置加载
            
        Returns:
            BaseModelProvider: 模型提供商实例
            
        Raises:
            ValueError: 模型配置无效
            RuntimeError: 模型加载失败
        """
        # 如果模型已加载，直接返回
        if model_id in self._providers:
            logger.info(f"模型已加载: {model_id}")
            return self._providers[model_id]
            
        # 获取模型配置
        if config is None:
            config = self._get_model_config(model_id)
            
        if config is None:
            raise ValueError(f"未找到模型配置: {model_id}")
            
        # 检查提供商类是否注册
        if (config.provider not in self._provider_classes or 
            config.model_type not in self._provider_classes[config.provider]):
            raise ValueError(f"未注册的提供商: {config.provider.value}/{config.model_type.value}")
            
        # 创建提供商实例
        provider_class = self._provider_classes[config.provider][config.model_type]
        provider = provider_class(config)
        
        try:
            # 加载模型
            provider.load_model()
            
            # 缓存提供商实例
            self._providers[model_id] = provider
            
            logger.info(f"模型加载成功: {model_id}")
            return provider
            
        except Exception as e:
            logger.error(f"模型加载失败: {model_id}, 错误: {e}")
            raise RuntimeError(f"模型加载失败: {e}")
            
    def unload_model(self, model_id: str) -> None:
        """卸载模型
        
        Args:
            model_id: 模型ID
        """
        if model_id in self._providers:
            provider = self._providers[model_id]
            provider.unload_model()
            del self._providers[model_id]
            
            # 更新当前模型引用
            if self._current_embedding_model == model_id:
                self._current_embedding_model = None
            if self._current_chat_model == model_id:
                self._current_chat_model = None
                
            logger.info(f"模型已卸载: {model_id}")
        else:
            logger.warning(f"尝试卸载未加载的模型: {model_id}")
            
    def switch_embedding_model(self, model_id: str) -> BaseEmbeddingProvider:
        """切换嵌入模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            BaseEmbeddingProvider: 嵌入模型提供商
        """
        provider = self.load_model(model_id)
        
        if not isinstance(provider, BaseEmbeddingProvider):
            raise ValueError(f"模型 {model_id} 不是嵌入模型")
            
        self._current_embedding_model = model_id
        logger.info(f"切换嵌入模型至: {model_id}")
        return provider
        
    def switch_chat_model(self, model_id: str) -> BaseChatProvider:
        """切换聊天模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            BaseChatProvider: 聊天模型提供商
        """
        provider = self.load_model(model_id)
        
        if not isinstance(provider, BaseChatProvider):
            raise ValueError(f"模型 {model_id} 不是聊天模型")
            
        self._current_chat_model = model_id
        logger.info(f"切换聊天模型至: {model_id}")
        return provider
        
    def get_current_embedding_model(self) -> Optional[BaseEmbeddingProvider]:
        """获取当前嵌入模型"""
        if self._current_embedding_model and self._current_embedding_model in self._providers:
            return self._providers[self._current_embedding_model]
        return None
        
    def get_current_chat_model(self) -> Optional[BaseChatProvider]:
        """获取当前聊天模型"""
        if self._current_chat_model and self._current_chat_model in self._providers:
            return self._providers[self._current_chat_model]
        return None
        
    def get_provider(self, model_type: ModelType) -> Optional[Union[BaseEmbeddingProvider, BaseChatProvider]]:
        """根据模型类型获取当前提供商
        
        Args:
            model_type: 模型类型
            
        Returns:
            Union[BaseEmbeddingProvider, BaseChatProvider]: 对应类型的提供商，如果未找到则返回None
        """
        if model_type == ModelType.EMBEDDING:
            return self.get_current_embedding_model()
        elif model_type == ModelType.CHAT:
            return self.get_current_chat_model()
        else:
            return None
        
    def get_loaded_models(self) -> List[str]:
        """获取已加载的模型列表"""
        return list(self._providers.keys())
        
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型信息
        
        Args:
            model_id: 模型ID
            
        Returns:
            Dict[str, Any]: 模型信息，如果模型未加载则返回None
        """
        if model_id in self._providers:
            return self._providers[model_id].get_model_info()
        return None
        
    def list_available_models(self, model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """列出可用的模型
        
        Args:
            model_type: 模型类型过滤，None表示所有类型
            
        Returns:
            List[Dict[str, Any]]: 模型信息列表
        """
        models = []
        
        # 从数据库获取模型配置
        db_configs = self._get_all_model_configs()
        
        # 合并默认配置
        all_configs = {**DEFAULT_MODEL_CONFIGS, **db_configs}
        
        for model_id, config in all_configs.items():
            if model_type is None or config.model_type == model_type:
                model_info = {
                    "model_id": model_id,
                    "name": config.name,
                    "provider": config.provider.value,
                    "type": config.model_type.value,
                    "is_active": config.is_active,
                    "is_available": config.is_available,
                    "is_loaded": model_id in self._providers
                }
                
                # 如果模型已加载，添加运行时信息
                if model_id in self._providers:
                    runtime_info = self._providers[model_id].get_model_info()
                    model_info.update(runtime_info)
                    
                models.append(model_info)
                
        return models
        
    def save_model_config(self, config: ModelConfig) -> None:
        """保存模型配置到数据库
        
        Args:
            config: 模型配置
        """
        db = next(get_session())
        try:
            # 查找现有配置
            existing = db.query(ModelConfig).filter(
                ModelConfig.model_id == config.model_id
            ).first()
            
            if existing:
                # 更新现有配置
                existing.name = config.name
                existing.provider = config.provider.value
                existing.model_type = config.model_type.value
                existing.config = config.config
                existing.is_active = config.is_active
                existing.is_available = config.is_available
                existing.updated_at = datetime.utcnow()
            else:
                # 创建新配置
                db_config = ModelConfig(
                    model_id=config.model_id,
                    name=config.name,
                    provider=config.provider.value,
                    model_type=config.model_type.value,
                    config=config.config,
                    is_active=config.is_active,
                    is_available=config.is_available
                )
                db.add(db_config)
                
            db.commit()
            logger.info(f"模型配置已保存: {config.model_id}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"保存模型配置失败: {e}")
            raise
        finally:
            db.close()
            
    def _get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """获取模型配置
        
        Args:
            model_id: 模型ID
            
        Returns:
            ModelConfig: 模型配置，如果未找到则返回None
        """
        # 首先检查默认配置
        if model_id in DEFAULT_MODEL_CONFIGS:
            return DEFAULT_MODEL_CONFIGS[model_id]
            
        # 从数据库查找
        db = next(get_session())
        try:
            db_config = db.query(ModelConfig).filter(
                ModelConfig.model_id == model_id
            ).first()
            
            if db_config:
                return ModelConfig(
                    model_id=db_config.model_id,
                    name=db_config.name,
                    provider=ModelProvider(db_config.provider),
                    model_type=ModelType(db_config.model_type),
                    config=db_config.config or {},
                    is_active=db_config.is_active,
                    is_available=db_config.is_available
                )
                
        except Exception as e:
            logger.error(f"获取模型配置失败: {e}")
        finally:
            db.close()
            
        return None
        
    def _get_all_model_configs(self) -> Dict[str, ModelConfig]:
        """获取所有数据库中的模型配置"""
        configs = {}
        db = next(get_session())
        try:
            db_configs = db.query(ModelConfig).all()
            
            for db_config in db_configs:
                config = ModelConfig(
                    model_id=db_config.model_id,
                    name=db_config.name,
                    provider=ModelProvider(db_config.provider),
                    model_type=ModelType(db_config.model_type),
                    config=db_config.config or {},
                    is_active=db_config.is_active,
                    is_available=db_config.is_available
                )
                configs[db_config.model_id] = config
                
        except Exception as e:
            logger.error(f"获取所有模型配置失败: {e}")
        finally:
            db.close()
            
        return configs
        
    def cleanup(self) -> None:
        """清理资源，卸载所有模型"""
        model_ids = list(self._providers.keys())
        for model_id in model_ids:
            self.unload_model(model_id)
        logger.info("模型管理器清理完成")


# 全局模型管理器实例
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """获取全局模型管理器实例"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def cleanup_model_manager() -> None:
    """清理全局模型管理器"""
    global _model_manager
    if _model_manager is not None:
        _model_manager.cleanup()
        _model_manager = None