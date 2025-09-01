"""文本预处理器模块

提供文本预处理功能，包括空白归一、Unicode标准化、
全角转半角、页眉页脚清理、去重等功能。
"""

import re
import unicodedata
from typing import Optional, Set
from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    """预处理配置"""
    normalize_whitespace: bool = True  # 空白字符归一化
    normalize_unicode: bool = True     # Unicode NFKC标准化
    convert_fullwidth: bool = True     # 全角转半角
    remove_urls_emails: bool = False   # 移除URL和邮箱
    remove_headers_footers: bool = False  # 移除页眉页脚
    remove_duplicates: bool = False    # 去重（基于simhash）
    min_line_length: int = 3          # 最小行长度
    max_line_length: int = 10000      # 最大行长度


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        
        # 预编译正则表达式
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self._whitespace_pattern = re.compile(r'\s+')
        self._multiple_newlines = re.compile(r'\n\s*\n\s*\n+')
        
        # 页眉页脚常见模式
        self._header_footer_patterns = [
            re.compile(r'^\s*第\s*\d+\s*页\s*$', re.MULTILINE),  # 页码
            re.compile(r'^\s*Page\s+\d+\s*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\s*\d+\s*/\s*\d+\s*$', re.MULTILINE),  # 页码格式 1/10
            re.compile(r'^\s*-\s*\d+\s*-\s*$', re.MULTILINE),   # -1-格式页码
        ]
    
    def preprocess(self, text: str) -> str:
        """执行文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            str: 预处理后的文本
        """
        if not text or not text.strip():
            return ""
        
        result = text
        
        # 1. Unicode标准化
        if self.config.normalize_unicode:
            result = self._normalize_unicode(result)
        
        # 2. 全角转半角
        if self.config.convert_fullwidth:
            result = self._convert_fullwidth_to_halfwidth(result)
        
        # 3. 移除URL和邮箱
        if self.config.remove_urls_emails:
            result = self._remove_urls_emails(result)
        
        # 4. 移除页眉页脚
        if self.config.remove_headers_footers:
            result = self._remove_headers_footers(result)
        
        # 5. 空白字符归一化
        if self.config.normalize_whitespace:
            result = self._normalize_whitespace(result)
        
        # 6. 清理过长或过短的行
        result = self._clean_lines(result)
        
        return result.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """Unicode NFKC标准化
        
        NFKC: 标准分解后再标准合成，同时进行兼容性分解
        可以处理全角/半角、上标下标等兼容性字符
        """
        return unicodedata.normalize('NFKC', text)
    
    def _convert_fullwidth_to_halfwidth(self, text: str) -> str:
        """全角字符转半角字符
        
        主要处理全角数字、字母、标点符号
        """
        result = []
        for char in text:
            # 全角空格转半角空格
            if char == '\u3000':
                result.append(' ')
            # 全角字符转半角（ASCII范围）
            elif '\uff01' <= char <= '\uff5e':
                # 全角字符Unicode码点减去0xfee0得到对应半角字符
                result.append(chr(ord(char) - 0xfee0))
            else:
                result.append(char)
        
        return ''.join(result)
    
    def _remove_urls_emails(self, text: str) -> str:
        """移除URL和邮箱地址"""
        # 移除URL
        text = self._url_pattern.sub('', text)
        # 移除邮箱
        text = self._email_pattern.sub('', text)
        return text
    
    def _remove_headers_footers(self, text: str) -> str:
        """移除页眉页脚
        
        识别并移除常见的页眉页脚模式，如页码、版权信息等
        """
        result = text
        
        # 应用预定义的页眉页脚模式
        for pattern in self._header_footer_patterns:
            result = pattern.sub('', result)
        
        # 移除文档开头和结尾的常见模式
        lines = result.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # 跳过可能的页眉页脚行
            if self._is_likely_header_footer(line_stripped):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _is_likely_header_footer(self, line: str) -> bool:
        """判断是否可能是页眉页脚"""
        if not line:
            return False
        
        # 纯数字行（可能是页码）
        if line.isdigit() and len(line) <= 3:
            return True
        
        # 包含版权符号
        if '©' in line or 'copyright' in line.lower():
            return True
        
        # 很短的行且包含特殊字符
        if len(line) <= 10 and any(char in line for char in '-_=*'):
            return True
        
        return False
    
    def _normalize_whitespace(self, text: str) -> str:
        """空白字符归一化
        
        1. 将制表符、多个空格等归一为单个空格
        2. 规范化换行符
        3. 移除行首行尾空白
        4. 合并多个连续换行
        """
        # 将制表符转换为空格
        text = text.replace('\t', ' ')
        
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 处理每一行
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 移除行首行尾空白
            line = line.strip()
            # 将多个空格合并为单个空格
            line = self._whitespace_pattern.sub(' ', line)
            cleaned_lines.append(line)
        
        # 重新组合
        result = '\n'.join(cleaned_lines)
        
        # 合并多个连续换行为最多两个换行（保持段落分隔）
        result = self._multiple_newlines.sub('\n\n', result)
        
        return result
    
    def _clean_lines(self, text: str) -> str:
        """清理过长或过短的行"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # 跳过过短的行（但保留空行用于段落分隔）
            if line_stripped and len(line_stripped) < self.config.min_line_length:
                continue
            
            # 截断过长的行
            if len(line) > self.config.max_line_length:
                line = line[:self.config.max_line_length] + '...'
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def remove_duplicates(self, texts: list[str]) -> list[str]:
        """基于内容相似度的去重
        
        使用简单的字符串相似度进行去重，
        后续可以升级为simhash等更高级的算法
        
        Args:
            texts: 文本列表
            
        Returns:
            list[str]: 去重后的文本列表
        """
        if not self.config.remove_duplicates:
            return texts
        
        unique_texts = []
        seen_hashes = set()
        
        for text in texts:
            # 使用简单的哈希去重
            text_hash = hash(text.strip().lower())
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)
        
        return unique_texts
    
    def get_text_stats(self, text: str) -> dict:
        """获取文本统计信息
        
        Args:
            text: 文本内容
            
        Returns:
            dict: 统计信息
        """
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        return {
            'total_chars': len(text),
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'avg_line_length': sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0,
            'max_line_length': max(len(line) for line in lines) if lines else 0,
            'whitespace_ratio': (len(text) - len(text.replace(' ', '').replace('\n', '').replace('\t', ''))) / len(text) if text else 0
        }


def create_preprocessor(config_dict: Optional[dict] = None) -> TextPreprocessor:
    """创建预处理器实例
    
    Args:
        config_dict: 配置字典
        
    Returns:
        TextPreprocessor: 预处理器实例
    """
    if config_dict:
        config = PreprocessConfig(**config_dict)
    else:
        config = PreprocessConfig()
    
    return TextPreprocessor(config)