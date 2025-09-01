"""OpenAI聊天模型提供商

实现OpenAI聊天模型的接口
"""

import logging
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator
import openai
from openai import OpenAI

from .base import BaseChatProvider
from ..config import ModelConfig

logger = logging.getLogger(__name__)


class OpenAIChatProvider(BaseChatProvider):
    """OpenAI聊天模型提供商"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client: Optional[OpenAI] = None
        
    def load_model(self) -> None:
        """加载OpenAI客户端"""
        try:
            api_key = self.config.config.get('api_key')
            base_url = self.config.config.get('base_url')
            
            if not api_key:
                raise ValueError("OpenAI API密钥未配置")
                
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            # 测试连接
            self.client.models.list()
            
            self._is_loaded = True
            logger.info(f"OpenAI聊天模型加载成功: {self.config.model_id}")
            
        except Exception as e:
            logger.error(f"OpenAI聊天模型加载失败: {e}")
            raise
            
    def unload_model(self) -> None:
        """卸载模型"""
        if self.client:
            self.client = None
        self._is_loaded = False
        logger.info(f"OpenAI聊天模型已卸载: {self.config.model_id}")
        
    def is_available(self) -> bool:
        """检查模型是否可用"""
        if not self.client:
            return False
            
        try:
            # 尝试列出模型来测试连接
            self.client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI聊天模型不可用: {e}")
            return False
            
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """聊天对话
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
            **kwargs: 其他参数
            
        Returns:
            str: 回复内容
        """
        if not self.client:
            raise RuntimeError("模型未加载")
            
        try:
            # 合并配置参数
            chat_params = {
                'model': self.config.model_id,
                'messages': messages,
                'temperature': kwargs.get('temperature', self.config.config.get('temperature', 0.7)),
                'max_tokens': kwargs.get('max_tokens', self.config.config.get('max_tokens', 1000)),
                'top_p': kwargs.get('top_p', self.config.config.get('top_p', 1.0)),
                'frequency_penalty': kwargs.get('frequency_penalty', self.config.config.get('frequency_penalty', 0.0)),
                'presence_penalty': kwargs.get('presence_penalty', self.config.config.get('presence_penalty', 0.0))
            }
            
            response = self.client.chat.completions.create(**chat_params)
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            else:
                raise RuntimeError("OpenAI返回空响应")
                
        except Exception as e:
            logger.error(f"OpenAI聊天失败: {e}")
            raise
            
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """流式聊天对话
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Yields:
            str: 流式回复内容片段
        """
        if not self.client:
            raise RuntimeError("模型未加载")
            
        try:
            # 合并配置参数
            chat_params = {
                'model': self.config.model_id,
                'messages': messages,
                'stream': True,
                'temperature': kwargs.get('temperature', self.config.config.get('temperature', 0.7)),
                'max_tokens': kwargs.get('max_tokens', self.config.config.get('max_tokens', 1000)),
                'top_p': kwargs.get('top_p', self.config.config.get('top_p', 1.0)),
                'frequency_penalty': kwargs.get('frequency_penalty', self.config.config.get('frequency_penalty', 0.0)),
                'presence_penalty': kwargs.get('presence_penalty', self.config.config.get('presence_penalty', 0.0))
            }
            
            stream = self.client.chat.completions.create(**chat_params)
            
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                        
        except Exception as e:
            logger.error(f"OpenAI流式聊天失败: {e}")
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        
        if self.client:
            try:
                # 获取模型详细信息
                model_info = self.client.models.retrieve(self.config.model_id)
                info.update({
                    'model_details': {
                        'id': model_info.id,
                        'created': model_info.created,
                        'owned_by': model_info.owned_by
                    }
                })
            except Exception as e:
                logger.warning(f"获取OpenAI模型详细信息失败: {e}")
                
        return info