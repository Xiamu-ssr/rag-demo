"""文本分段器模块

实现父子分段模型，支持flat和parent_child两种分段模式。
提供基于规则和语义的文本分段功能。
"""

import re
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SplitMode(Enum):
    """分段模式枚举"""
    FLAT = "flat"                    # 平铺模式：所有分段平等
    PARENT_CHILD = "parent_child"    # 父子模式：大分段包含小分段


@dataclass
class ChunkMetadata:
    """分段元数据"""
    chunk_id: str                    # 分段ID
    parent_id: Optional[str] = None  # 父分段ID（仅parent_child模式）
    level: int = 0                   # 分段层级（0为顶级）
    start_pos: int = 0               # 在原文中的起始位置
    end_pos: int = 0                 # 在原文中的结束位置
    token_count: int = 0             # token数量
    char_count: int = 0              # 字符数量
    section_title: Optional[str] = None  # 章节标题
    

@dataclass
class TextChunk:
    """文本分段"""
    content: str                     # 分段内容
    metadata: ChunkMetadata          # 元数据
    
    def __len__(self) -> int:
        return len(self.content)
    
    def __str__(self) -> str:
        return f"Chunk({self.metadata.chunk_id}): {self.content[:50]}..."


@dataclass
class SplitterConfig:
    """分段器配置"""
    # 基础配置
    chunk_size: int = 1000           # 目标分段大小（字符数）
    chunk_overlap: int = 200         # 分段重叠大小
    min_chunk_size: int = 100        # 最小分段大小
    max_chunk_size: int = 2000       # 最大分段大小
    
    # 模式配置
    split_mode: SplitMode = SplitMode.FLAT
    
    # 父子模式配置
    parent_chunk_size: int = 2000    # 父分段大小
    child_chunk_size: int = 500      # 子分段大小
    parent_overlap: int = 100        # 父分段重叠
    child_overlap: int = 50          # 子分段重叠
    
    # 分割策略
    respect_sentence_boundary: bool = True   # 尊重句子边界
    respect_paragraph_boundary: bool = True  # 尊重段落边界
    respect_section_boundary: bool = True    # 尊重章节边界
    
    # 语言相关
    language: str = "zh"             # 语言代码
    

class BaseSplitter(ABC):
    """分段器基类"""
    
    def __init__(self, config: Optional[SplitterConfig] = None):
        self.config = config or SplitterConfig()
        
        # 预编译正则表达式
        self._sentence_endings = re.compile(r'[.!?。！？]\s*')
        self._paragraph_separator = re.compile(r'\n\s*\n')
        self._section_headers = re.compile(
            r'^(#{1,6}\s+.+|第[一二三四五六七八九十\d]+[章节部分]\s*.+|\d+\.\s*.+)$',
            re.MULTILINE
        )
    
    @abstractmethod
    def split(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """分割文本
        
        Args:
            text: 待分割的文本
            metadata: 文档元数据
            
        Returns:
            List[TextChunk]: 分段列表
        """
        pass
    
    def _estimate_token_count(self, text: str) -> int:
        """估算token数量
        
        简单估算：中文按字符数，英文按单词数*1.3
        """
        if not text:
            return 0
        
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 统计英文单词
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        
        # 估算token数：中文字符*1 + 英文单词*1.3
        return chinese_chars + int(english_words * 1.3)
    
    def _find_best_split_point(self, text: str, target_pos: int) -> int:
        """找到最佳分割点
        
        优先级：段落边界 > 句子边界 > 空格 > 目标位置
        """
        if target_pos >= len(text):
            return len(text)
        
        # 搜索范围：目标位置前后的一定范围
        search_range = min(100, len(text) // 10)
        start = max(0, target_pos - search_range)
        end = min(len(text), target_pos + search_range)
        
        search_text = text[start:end]
        
        # 1. 优先寻找段落边界
        if self.config.respect_paragraph_boundary:
            paragraph_matches = list(self._paragraph_separator.finditer(search_text))
            if paragraph_matches:
                # 找到最接近目标位置的段落边界
                best_match = min(paragraph_matches, 
                                key=lambda m: abs(m.end() + start - target_pos))
                return best_match.end() + start
        
        # 2. 寻找句子边界
        if self.config.respect_sentence_boundary:
            sentence_matches = list(self._sentence_endings.finditer(search_text))
            if sentence_matches:
                best_match = min(sentence_matches,
                                key=lambda m: abs(m.end() + start - target_pos))
                return best_match.end() + start
        
        # 3. 寻找空格
        for offset in range(search_range):
            # 向前搜索
            if target_pos - offset >= 0 and text[target_pos - offset].isspace():
                return target_pos - offset
            # 向后搜索
            if target_pos + offset < len(text) and text[target_pos + offset].isspace():
                return target_pos + offset
        
        # 4. 返回目标位置
        return target_pos
    
    def _extract_section_info(self, text: str, start_pos: int) -> Optional[str]:
        """提取章节信息"""
        if not self.config.respect_section_boundary:
            return None
        
        # 在分段开始位置向前搜索最近的章节标题
        text_before = text[:start_pos]
        matches = list(self._section_headers.finditer(text_before))
        
        if matches:
            last_match = matches[-1]
            return last_match.group().strip()
        
        return None


class FlatSplitter(BaseSplitter):
    """平铺分段器
    
    将文本分割为大小相近的平等分段
    """
    
    def split(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """执行平铺分段"""
        if not text or not text.strip():
            return []
        
        chunks = []
        current_pos = 0
        chunk_id = 0
        
        while current_pos < len(text):
            # 计算分段结束位置
            end_pos = min(current_pos + self.config.chunk_size, len(text))
            
            # 找到最佳分割点
            if end_pos < len(text):
                end_pos = self._find_best_split_point(text, end_pos)
            
            # 提取分段内容
            chunk_content = text[current_pos:end_pos].strip()
            
            if len(chunk_content) >= self.config.min_chunk_size:
                # 创建分段元数据
                chunk_metadata = ChunkMetadata(
                    chunk_id=f"chunk_{chunk_id}",
                    start_pos=current_pos,
                    end_pos=end_pos,
                    char_count=len(chunk_content),
                    token_count=self._estimate_token_count(chunk_content),
                    section_title=self._extract_section_info(text, current_pos)
                )
                
                # 创建分段
                chunk = TextChunk(content=chunk_content, metadata=chunk_metadata)
                chunks.append(chunk)
                chunk_id += 1
            
            # 计算下一个分段的起始位置（考虑重叠）
            next_pos = end_pos - self.config.chunk_overlap
            current_pos = max(current_pos + 1, next_pos)  # 确保前进
        
        return chunks


class ParentChildSplitter(BaseSplitter):
    """父子分段器
    
    创建大分段（父）和小分段（子）的层次结构
    """
    
    def split(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """执行父子分段"""
        if not text or not text.strip():
            return []
        
        # 首先创建父分段
        parent_chunks = self._create_parent_chunks(text)
        
        # 然后为每个父分段创建子分段
        all_chunks = []
        
        for parent_chunk in parent_chunks:
            all_chunks.append(parent_chunk)
            
            # 创建子分段
            child_chunks = self._create_child_chunks(
                parent_chunk.content, 
                parent_chunk.metadata.chunk_id,
                parent_chunk.metadata.start_pos
            )
            all_chunks.extend(child_chunks)
        
        return all_chunks
    
    def _create_parent_chunks(self, text: str) -> List[TextChunk]:
        """创建父分段"""
        chunks = []
        current_pos = 0
        parent_id = 0
        
        while current_pos < len(text):
            # 计算父分段结束位置
            end_pos = min(current_pos + self.config.parent_chunk_size, len(text))
            
            # 找到最佳分割点
            if end_pos < len(text):
                end_pos = self._find_best_split_point(text, end_pos)
            
            # 提取分段内容
            chunk_content = text[current_pos:end_pos].strip()
            
            if len(chunk_content) >= self.config.min_chunk_size:
                # 创建父分段元数据
                chunk_metadata = ChunkMetadata(
                    chunk_id=f"parent_{parent_id}",
                    level=0,
                    start_pos=current_pos,
                    end_pos=end_pos,
                    char_count=len(chunk_content),
                    token_count=self._estimate_token_count(chunk_content),
                    section_title=self._extract_section_info(text, current_pos)
                )
                
                # 创建父分段
                chunk = TextChunk(content=chunk_content, metadata=chunk_metadata)
                chunks.append(chunk)
                parent_id += 1
            
            # 计算下一个父分段的起始位置
            next_pos = end_pos - self.config.parent_overlap
            current_pos = max(current_pos + 1, next_pos)
        
        return chunks
    
    def _create_child_chunks(self, parent_text: str, parent_id: str, parent_start_pos: int) -> List[TextChunk]:
        """为父分段创建子分段"""
        chunks = []
        current_pos = 0
        child_id = 0
        
        while current_pos < len(parent_text):
            # 计算子分段结束位置
            end_pos = min(current_pos + self.config.child_chunk_size, len(parent_text))
            
            # 找到最佳分割点
            if end_pos < len(parent_text):
                end_pos = self._find_best_split_point(parent_text, end_pos)
            
            # 提取分段内容
            chunk_content = parent_text[current_pos:end_pos].strip()
            
            if len(chunk_content) >= self.config.min_chunk_size:
                # 创建子分段元数据
                chunk_metadata = ChunkMetadata(
                    chunk_id=f"{parent_id}_child_{child_id}",
                    parent_id=parent_id,
                    level=1,
                    start_pos=parent_start_pos + current_pos,
                    end_pos=parent_start_pos + end_pos,
                    char_count=len(chunk_content),
                    token_count=self._estimate_token_count(chunk_content)
                )
                
                # 创建子分段
                chunk = TextChunk(content=chunk_content, metadata=chunk_metadata)
                chunks.append(chunk)
                child_id += 1
            
            # 计算下一个子分段的起始位置
            next_pos = end_pos - self.config.child_overlap
            current_pos = max(current_pos + 1, next_pos)
        
        return chunks


class TextSplitter:
    """文本分段器主类
    
    根据配置选择合适的分段策略
    """
    
    def __init__(self, config: Optional[SplitterConfig] = None):
        self.config = config or SplitterConfig()
        
        # 根据模式选择分段器
        if self.config.split_mode == SplitMode.FLAT:
            self._splitter = FlatSplitter(self.config)
        elif self.config.split_mode == SplitMode.PARENT_CHILD:
            self._splitter = ParentChildSplitter(self.config)
        else:
            raise ValueError(f"Unsupported split mode: {self.config.split_mode}")
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """分割文本
        
        Args:
            text: 待分割的文本
            metadata: 文档元数据
            
        Returns:
            List[TextChunk]: 分段列表
        """
        return self._splitter.split(text, metadata)
    
    def get_chunks_by_level(self, chunks: List[TextChunk], level: int = 0) -> List[TextChunk]:
        """按层级获取分段
        
        Args:
            chunks: 分段列表
            level: 目标层级（0为父分段，1为子分段）
            
        Returns:
            List[TextChunk]: 指定层级的分段列表
        """
        return [chunk for chunk in chunks if chunk.metadata.level == level]
    
    def get_parent_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """获取所有父分段"""
        return self.get_chunks_by_level(chunks, level=0)
    
    def get_child_chunks(self, chunks: List[TextChunk], parent_id: Optional[str] = None) -> List[TextChunk]:
        """获取子分段
        
        Args:
            chunks: 分段列表
            parent_id: 父分段ID，如果为None则返回所有子分段
            
        Returns:
            List[TextChunk]: 子分段列表
        """
        child_chunks = self.get_chunks_by_level(chunks, level=1)
        
        if parent_id:
            return [chunk for chunk in child_chunks if chunk.metadata.parent_id == parent_id]
        
        return child_chunks
    
    def get_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """获取分段统计信息
        
        Args:
            chunks: 分段列表
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not chunks:
            return {}
        
        char_counts = [chunk.metadata.char_count for chunk in chunks]
        token_counts = [chunk.metadata.token_count for chunk in chunks]
        
        parent_chunks = self.get_parent_chunks(chunks)
        child_chunks = self.get_child_chunks(chunks)
        
        return {
            'total_chunks': len(chunks),
            'parent_chunks': len(parent_chunks),
            'child_chunks': len(child_chunks),
            'avg_char_count': sum(char_counts) / len(char_counts),
            'avg_token_count': sum(token_counts) / len(token_counts),
            'max_char_count': max(char_counts),
            'min_char_count': min(char_counts),
            'total_chars': sum(char_counts),
            'total_tokens': sum(token_counts)
        }


def create_splitter(config_dict: Optional[Dict[str, Any]] = None) -> TextSplitter:
    """创建分段器实例
    
    Args:
        config_dict: 配置字典
        
    Returns:
        TextSplitter: 分段器实例
    """
    if config_dict:
        # 处理枚举类型
        if 'split_mode' in config_dict and isinstance(config_dict['split_mode'], str):
            config_dict['split_mode'] = SplitMode(config_dict['split_mode'])
        
        config = SplitterConfig(**config_dict)
    else:
        config = SplitterConfig()
    
    return TextSplitter(config)