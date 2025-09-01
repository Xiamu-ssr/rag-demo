"""Token计数工具模块

提供精确的token计数功能，支持多种tokenizer和语言模型。
包括中文、英文等多语言的token计算。
"""

import re
import math
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TokenizerType(Enum):
    """Tokenizer类型枚举"""
    SIMPLE = "simple"                # 简单估算
    TIKTOKEN = "tiktoken"            # OpenAI tiktoken
    HUGGINGFACE = "huggingface"      # HuggingFace tokenizer
    SENTENCEPIECE = "sentencepiece"  # SentencePiece tokenizer


@dataclass
class TokenStats:
    """Token统计信息"""
    token_count: int                 # token数量
    char_count: int                  # 字符数量
    word_count: int                  # 单词数量
    chinese_char_count: int          # 中文字符数量
    english_word_count: int          # 英文单词数量
    punctuation_count: int           # 标点符号数量
    whitespace_count: int            # 空白字符数量
    
    @property
    def chars_per_token(self) -> float:
        """每个token的平均字符数"""
        return self.char_count / self.token_count if self.token_count > 0 else 0
    
    @property
    def tokens_per_word(self) -> float:
        """每个单词的平均token数"""
        return self.token_count / self.word_count if self.word_count > 0 else 0


class BaseTokenCounter:
    """Token计数器基类"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "default"
        
        # 预编译正则表达式
        self._chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        self._english_word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        self._punctuation_pattern = re.compile(r'[^\w\s]')
        self._whitespace_pattern = re.compile(r'\s')
    
    def count_tokens(self, text: str) -> int:
        """计算token数量
        
        Args:
            text: 输入文本
            
        Returns:
            int: token数量
        """
        raise NotImplementedError
    
    def get_token_stats(self, text: str) -> TokenStats:
        """获取详细的token统计信息
        
        Args:
            text: 输入文本
            
        Returns:
            TokenStats: 统计信息
        """
        if not text:
            return TokenStats(
                token_count=0, char_count=0, word_count=0,
                chinese_char_count=0, english_word_count=0,
                punctuation_count=0, whitespace_count=0
            )
        
        # 基础统计
        char_count = len(text)
        chinese_chars = self._chinese_pattern.findall(text)
        english_words = self._english_word_pattern.findall(text)
        punctuation = self._punctuation_pattern.findall(text)
        whitespace = self._whitespace_pattern.findall(text)
        
        chinese_char_count = len(chinese_chars)
        english_word_count = len(english_words)
        punctuation_count = len(punctuation)
        whitespace_count = len(whitespace)
        
        # 总单词数（中文字符 + 英文单词）
        word_count = chinese_char_count + english_word_count
        
        # 计算token数量
        token_count = self.count_tokens(text)
        
        return TokenStats(
            token_count=token_count,
            char_count=char_count,
            word_count=word_count,
            chinese_char_count=chinese_char_count,
            english_word_count=english_word_count,
            punctuation_count=punctuation_count,
            whitespace_count=whitespace_count
        )
    
    def estimate_cost(self, text: str, price_per_1k_tokens: float) -> float:
        """估算处理成本
        
        Args:
            text: 输入文本
            price_per_1k_tokens: 每1000个token的价格
            
        Returns:
            float: 估算成本
        """
        token_count = self.count_tokens(text)
        return (token_count / 1000) * price_per_1k_tokens


class SimpleTokenCounter(BaseTokenCounter):
    """简单Token计数器
    
    基于启发式规则进行token估算，不依赖外部库
    """
    
    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name)
        
        # 不同语言的token比率（经验值）
        self.token_ratios = {
            'chinese': 1.0,      # 中文字符：1字符 ≈ 1token
            'english': 1.3,      # 英文单词：1单词 ≈ 1.3token
            'punctuation': 0.5,  # 标点符号：1符号 ≈ 0.5token
            'number': 0.8,       # 数字：1数字 ≈ 0.8token
        }
    
    def count_tokens(self, text: str) -> int:
        """基于启发式规则估算token数量"""
        if not text:
            return 0
        
        # 统计各类字符
        chinese_chars = len(self._chinese_pattern.findall(text))
        english_words = len(self._english_word_pattern.findall(text))
        punctuation = len(self._punctuation_pattern.findall(text))
        
        # 统计数字
        numbers = len(re.findall(r'\d+', text))
        
        # 计算token数量
        token_count = (
            chinese_chars * self.token_ratios['chinese'] +
            english_words * self.token_ratios['english'] +
            punctuation * self.token_ratios['punctuation'] +
            numbers * self.token_ratios['number']
        )
        
        return max(1, int(token_count))


class TikTokenCounter(BaseTokenCounter):
    """TikToken计数器
    
    使用OpenAI的tiktoken库进行精确计数
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__(model_name)
        
        try:
            import tiktoken
            self.tiktoken = tiktoken
            
            # 根据模型名称获取编码器
            if model_name in ["gpt-4", "gpt-4-32k", "gpt-4-turbo", "gpt-4o"]:
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            elif model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"]:
                self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # 默认使用cl100k_base编码
                self.encoding = tiktoken.get_encoding("cl100k_base")
                
        except ImportError:
            logger.warning("tiktoken库未安装，将使用简单估算")
            self.tiktoken = None
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """使用tiktoken精确计算token数量"""
        if not text:
            return 0
        
        if self.encoding is None:
            # 回退到简单估算
            simple_counter = SimpleTokenCounter()
            return simple_counter.count_tokens(text)
        
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"tiktoken编码失败: {e}，使用简单估算")
            simple_counter = SimpleTokenCounter()
            return simple_counter.count_tokens(text)


class HuggingFaceTokenCounter(BaseTokenCounter):
    """HuggingFace Tokenizer计数器
    
    使用HuggingFace transformers库的tokenizer
    """
    
    def __init__(self, model_name: str = "bert-base-chinese"):
        super().__init__(model_name)
        
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ImportError:
            logger.warning("transformers库未安装，将使用简单估算")
            self.tokenizer = None
        except Exception as e:
            logger.warning(f"加载tokenizer失败: {e}，将使用简单估算")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """使用HuggingFace tokenizer计算token数量"""
        if not text:
            return 0
        
        if self.tokenizer is None:
            # 回退到简单估算
            simple_counter = SimpleTokenCounter()
            return simple_counter.count_tokens(text)
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logger.warning(f"HuggingFace tokenizer编码失败: {e}，使用简单估算")
            simple_counter = SimpleTokenCounter()
            return simple_counter.count_tokens(text)


class TokenCounter:
    """统一的Token计数器
    
    根据配置选择合适的计数策略
    """
    
    def __init__(self, 
                 tokenizer_type: TokenizerType = TokenizerType.SIMPLE,
                 model_name: Optional[str] = None):
        self.tokenizer_type = tokenizer_type
        self.model_name = model_name
        
        # 创建具体的计数器
        self._counter = self._create_counter()
    
    def _create_counter(self) -> BaseTokenCounter:
        """创建具体的计数器实例"""
        if self.tokenizer_type == TokenizerType.SIMPLE:
            return SimpleTokenCounter(self.model_name)
        elif self.tokenizer_type == TokenizerType.TIKTOKEN:
            return TikTokenCounter(self.model_name or "gpt-3.5-turbo")
        elif self.tokenizer_type == TokenizerType.HUGGINGFACE:
            return HuggingFaceTokenCounter(self.model_name or "bert-base-chinese")
        else:
            logger.warning(f"不支持的tokenizer类型: {self.tokenizer_type}，使用简单估算")
            return SimpleTokenCounter(self.model_name)
    
    def count_tokens(self, text: str) -> int:
        """计算token数量"""
        return self._counter.count_tokens(text)
    
    def get_token_stats(self, text: str) -> TokenStats:
        """获取详细统计信息"""
        return self._counter.get_token_stats(text)
    
    def estimate_cost(self, text: str, price_per_1k_tokens: float) -> float:
        """估算处理成本"""
        return self._counter.estimate_cost(text, price_per_1k_tokens)
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """批量计算token数量
        
        Args:
            texts: 文本列表
            
        Returns:
            List[int]: token数量列表
        """
        return [self.count_tokens(text) for text in texts]
    
    def get_stats_batch(self, texts: List[str]) -> List[TokenStats]:
        """批量获取统计信息
        
        Args:
            texts: 文本列表
            
        Returns:
            List[TokenStats]: 统计信息列表
        """
        return [self.get_token_stats(text) for text in texts]
    
    def find_optimal_chunk_size(self, 
                               text: str, 
                               max_tokens: int, 
                               overlap_ratio: float = 0.1) -> Tuple[int, int]:
        """找到最优的分块大小
        
        Args:
            text: 输入文本
            max_tokens: 最大token数
            overlap_ratio: 重叠比例
            
        Returns:
            Tuple[int, int]: (chunk_size_chars, overlap_chars)
        """
        total_tokens = self.count_tokens(text)
        total_chars = len(text)
        
        if total_tokens <= max_tokens:
            return total_chars, 0
        
        # 估算字符到token的比例
        chars_per_token = total_chars / total_tokens
        
        # 计算分块大小（字符数）
        chunk_size_chars = int(max_tokens * chars_per_token)
        overlap_chars = int(chunk_size_chars * overlap_ratio)
        
        return chunk_size_chars, overlap_chars


# 便捷函数
def count_tokens(text: str, 
                tokenizer_type: TokenizerType = TokenizerType.SIMPLE,
                model_name: Optional[str] = None) -> int:
    """计算文本token数量的便捷函数
    
    Args:
        text: 输入文本
        tokenizer_type: tokenizer类型
        model_name: 模型名称
        
    Returns:
        int: token数量
    """
    counter = TokenCounter(tokenizer_type, model_name)
    return counter.count_tokens(text)


def get_token_stats(text: str,
                   tokenizer_type: TokenizerType = TokenizerType.SIMPLE,
                   model_name: Optional[str] = None) -> TokenStats:
    """获取文本token统计信息的便捷函数
    
    Args:
        text: 输入文本
        tokenizer_type: tokenizer类型
        model_name: 模型名称
        
    Returns:
        TokenStats: 统计信息
    """
    counter = TokenCounter(tokenizer_type, model_name)
    return counter.get_token_stats(text)


def estimate_processing_cost(text: str,
                           price_per_1k_tokens: float,
                           tokenizer_type: TokenizerType = TokenizerType.SIMPLE,
                           model_name: Optional[str] = None) -> float:
    """估算文本处理成本的便捷函数
    
    Args:
        text: 输入文本
        price_per_1k_tokens: 每1000个token的价格
        tokenizer_type: tokenizer类型
        model_name: 模型名称
        
    Returns:
        float: 估算成本
    """
    counter = TokenCounter(tokenizer_type, model_name)
    return counter.estimate_cost(text, price_per_1k_tokens)


# 预定义的计数器实例
default_counter = TokenCounter(TokenizerType.SIMPLE)
gpt_counter = TokenCounter(TokenizerType.TIKTOKEN, "gpt-3.5-turbo")
bert_counter = TokenCounter(TokenizerType.HUGGINGFACE, "bert-base-chinese")