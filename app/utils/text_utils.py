"""文本处理工具模块

提供常用的文本处理函数，包括文本清理、格式化、相似度计算等功能。
支持中英文混合文本的处理。
"""

import re
import unicodedata
import hashlib
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextMetrics:
    """文本度量信息"""
    length: int                    # 文本长度
    word_count: int               # 单词数量
    sentence_count: int           # 句子数量
    paragraph_count: int          # 段落数量
    chinese_char_count: int       # 中文字符数量
    english_word_count: int       # 英文单词数量
    punctuation_count: int        # 标点符号数量
    digit_count: int              # 数字字符数量
    whitespace_count: int         # 空白字符数量
    unique_words: int             # 唯一单词数量
    avg_word_length: float        # 平均单词长度
    avg_sentence_length: float    # 平均句子长度
    
    @property
    def lexical_diversity(self) -> float:
        """词汇多样性（唯一词汇数/总词汇数）"""
        return self.unique_words / self.word_count if self.word_count > 0 else 0
    
    @property
    def chinese_ratio(self) -> float:
        """中文字符比例"""
        return self.chinese_char_count / self.length if self.length > 0 else 0
    
    @property
    def english_ratio(self) -> float:
        """英文字符比例"""
        english_chars = sum(len(word) for word in range(self.english_word_count))
        return english_chars / self.length if self.length > 0 else 0


class TextCleaner:
    """文本清理器
    
    提供各种文本清理和标准化功能
    """
    
    def __init__(self):
        # 预编译正则表达式
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self._phone_pattern = re.compile(r'\b(?:\+?86)?1[3-9]\d{9}\b')
        self._chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        self._english_pattern = re.compile(r'[a-zA-Z]+')
        self._digit_pattern = re.compile(r'\d+')
        self._punctuation_pattern = re.compile(r'[^\w\s]')
        self._whitespace_pattern = re.compile(r'\s+')
        self._html_tag_pattern = re.compile(r'<[^>]+>')
        self._markdown_link_pattern = re.compile(r'\[([^\]]+)\]\([^\)]+\)')
        self._markdown_image_pattern = re.compile(r'!\[([^\]]*)\]\([^\)]+\)')
        
        # 全角到半角映射
        self._fullwidth_to_halfwidth = {
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
            'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
            'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
            'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
            'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
            'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y', 'Ｚ': 'Z',
            'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e',
            'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j',
            'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o',
            'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't',
            'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y', 'ｚ': 'z',
            '　': ' ', '，': ',', '。': '.', '；': ';', '：': ':',
            '？': '?', '！': '!', '（': '(', '）': ')', '【': '[',
            '】': ']', '「': '"', '」': '"', '『': "'", '』': "'"
        }
    
    def normalize_unicode(self, text: str) -> str:
        """Unicode标准化（NFKC）
        
        Args:
            text: 输入文本
            
        Returns:
            str: 标准化后的文本
        """
        return unicodedata.normalize('NFKC', text)
    
    def fullwidth_to_halfwidth(self, text: str) -> str:
        """全角转半角
        
        Args:
            text: 输入文本
            
        Returns:
            str: 转换后的文本
        """
        result = []
        for char in text:
            result.append(self._fullwidth_to_halfwidth.get(char, char))
        return ''.join(result)
    
    def normalize_whitespace(self, text: str) -> str:
        """标准化空白字符
        
        Args:
            text: 输入文本
            
        Returns:
            str: 标准化后的文本
        """
        # 将所有空白字符替换为单个空格
        text = self._whitespace_pattern.sub(' ', text)
        # 去除首尾空白
        return text.strip()
    
    def remove_urls(self, text: str) -> str:
        """移除URL
        
        Args:
            text: 输入文本
            
        Returns:
            str: 移除URL后的文本
        """
        return self._url_pattern.sub('', text)
    
    def remove_emails(self, text: str) -> str:
        """移除邮箱地址
        
        Args:
            text: 输入文本
            
        Returns:
            str: 移除邮箱后的文本
        """
        return self._email_pattern.sub('', text)
    
    def remove_phones(self, text: str) -> str:
        """移除电话号码
        
        Args:
            text: 输入文本
            
        Returns:
            str: 移除电话号码后的文本
        """
        return self._phone_pattern.sub('', text)
    
    def remove_html_tags(self, text: str) -> str:
        """移除HTML标签
        
        Args:
            text: 输入文本
            
        Returns:
            str: 移除HTML标签后的文本
        """
        return self._html_tag_pattern.sub('', text)
    
    def remove_markdown_links(self, text: str) -> str:
        """移除Markdown链接，保留链接文本
        
        Args:
            text: 输入文本
            
        Returns:
            str: 处理后的文本
        """
        # 替换链接为链接文本
        text = self._markdown_link_pattern.sub(r'\1', text)
        # 移除图片
        text = self._markdown_image_pattern.sub(r'\1', text)
        return text
    
    def remove_extra_punctuation(self, text: str) -> str:
        """移除多余的标点符号
        
        Args:
            text: 输入文本
            
        Returns:
            str: 处理后的文本
        """
        # 移除连续的标点符号
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[,]{2,}', ',', text)
        text = re.sub(r'[;]{2,}', ';', text)
        text = re.sub(r'[:]{2,}', ':', text)
        return text
    
    def clean_text(self, text: str, 
                   normalize_unicode: bool = True,
                   fullwidth_to_halfwidth: bool = True,
                   normalize_whitespace: bool = True,
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_phones: bool = False,
                   remove_html: bool = True,
                   remove_markdown: bool = True,
                   remove_extra_punct: bool = True) -> str:
        """综合文本清理
        
        Args:
            text: 输入文本
            normalize_unicode: 是否进行Unicode标准化
            fullwidth_to_halfwidth: 是否全角转半角
            normalize_whitespace: 是否标准化空白字符
            remove_urls: 是否移除URL
            remove_emails: 是否移除邮箱
            remove_phones: 是否移除电话号码
            remove_html: 是否移除HTML标签
            remove_markdown: 是否处理Markdown链接
            remove_extra_punct: 是否移除多余标点
            
        Returns:
            str: 清理后的文本
        """
        if not text:
            return text
        
        # 按顺序应用清理步骤
        if normalize_unicode:
            text = self.normalize_unicode(text)
        
        if fullwidth_to_halfwidth:
            text = self.fullwidth_to_halfwidth(text)
        
        if remove_html:
            text = self.remove_html_tags(text)
        
        if remove_markdown:
            text = self.remove_markdown_links(text)
        
        if remove_urls:
            text = self.remove_urls(text)
        
        if remove_emails:
            text = self.remove_emails(text)
        
        if remove_phones:
            text = self.remove_phones(text)
        
        if remove_extra_punct:
            text = self.remove_extra_punctuation(text)
        
        if normalize_whitespace:
            text = self.normalize_whitespace(text)
        
        return text


class TextAnalyzer:
    """文本分析器
    
    提供文本统计和分析功能
    """
    
    def __init__(self):
        # 预编译正则表达式
        self._chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        self._english_word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        self._sentence_pattern = re.compile(r'[.!?。！？]+\s*')
        self._paragraph_pattern = re.compile(r'\n\s*\n')
        self._digit_pattern = re.compile(r'\d')
        self._punctuation_pattern = re.compile(r'[^\w\s]')
        self._whitespace_pattern = re.compile(r'\s')
    
    def get_text_metrics(self, text: str) -> TextMetrics:
        """获取文本度量信息
        
        Args:
            text: 输入文本
            
        Returns:
            TextMetrics: 度量信息
        """
        if not text:
            return TextMetrics(
                length=0, word_count=0, sentence_count=0, paragraph_count=0,
                chinese_char_count=0, english_word_count=0, punctuation_count=0,
                digit_count=0, whitespace_count=0, unique_words=0,
                avg_word_length=0, avg_sentence_length=0
            )
        
        # 基础统计
        length = len(text)
        chinese_chars = self._chinese_pattern.findall(text)
        english_words = self._english_word_pattern.findall(text)
        sentences = self._sentence_pattern.split(text)
        paragraphs = self._paragraph_pattern.split(text)
        digits = self._digit_pattern.findall(text)
        punctuation = self._punctuation_pattern.findall(text)
        whitespace = self._whitespace_pattern.findall(text)
        
        chinese_char_count = len(chinese_chars)
        english_word_count = len(english_words)
        sentence_count = len([s for s in sentences if s.strip()])
        paragraph_count = len([p for p in paragraphs if p.strip()])
        digit_count = len(digits)
        punctuation_count = len(punctuation)
        whitespace_count = len(whitespace)
        
        # 总单词数（中文字符 + 英文单词）
        word_count = chinese_char_count + english_word_count
        
        # 唯一单词统计
        all_words = chinese_chars + english_words
        unique_words = len(set(all_words))
        
        # 平均长度计算
        avg_word_length = sum(len(word) for word in english_words) / len(english_words) if english_words else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return TextMetrics(
            length=length,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            chinese_char_count=chinese_char_count,
            english_word_count=english_word_count,
            punctuation_count=punctuation_count,
            digit_count=digit_count,
            whitespace_count=whitespace_count,
            unique_words=unique_words,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length
        )
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, int]]:
        """提取关键词（基于词频）
        
        Args:
            text: 输入文本
            top_k: 返回前k个关键词
            
        Returns:
            List[Tuple[str, int]]: (词, 频次)列表
        """
        if not text:
            return []
        
        # 提取中文字符和英文单词
        chinese_chars = self._chinese_pattern.findall(text.lower())
        english_words = self._english_word_pattern.findall(text.lower())
        
        # 合并所有词汇
        all_words = chinese_chars + english_words
        
        # 过滤停用词（简单版本）
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        filtered_words = [word for word in all_words if word not in stop_words and len(word) > 1]
        
        # 统计词频
        word_counts = Counter(filtered_words)
        
        return word_counts.most_common(top_k)
    
    def detect_language(self, text: str) -> str:
        """检测文本主要语言
        
        Args:
            text: 输入文本
            
        Returns:
            str: 语言类型 ('chinese', 'english', 'mixed')
        """
        if not text:
            return 'unknown'
        
        chinese_count = len(self._chinese_pattern.findall(text))
        english_count = len(self._english_word_pattern.findall(text))
        
        total_chars = chinese_count + english_count
        if total_chars == 0:
            return 'unknown'
        
        chinese_ratio = chinese_count / total_chars
        
        if chinese_ratio > 0.7:
            return 'chinese'
        elif chinese_ratio < 0.3:
            return 'english'
        else:
            return 'mixed'


class TextSimilarity:
    """文本相似度计算器"""
    
    @staticmethod
    def jaccard_similarity(text1: str, text2: str) -> float:
        """计算Jaccard相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            float: 相似度分数 (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # 转换为字符集合
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        # 计算交集和并集
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def cosine_similarity(text1: str, text2: str) -> float:
        """计算余弦相似度（基于字符频次）
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            float: 相似度分数 (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # 统计字符频次
        counter1 = Counter(text1.lower())
        counter2 = Counter(text2.lower())
        
        # 获取所有字符
        all_chars = set(counter1.keys()).union(set(counter2.keys()))
        
        # 构建向量
        vec1 = [counter1.get(char, 0) for char in all_chars]
        vec2 = [counter2.get(char, 0) for char in all_chars]
        
        # 计算点积和模长
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    @staticmethod
    def levenshtein_distance(text1: str, text2: str) -> int:
        """计算编辑距离
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            int: 编辑距离
        """
        if not text1:
            return len(text2)
        if not text2:
            return len(text1)
        
        # 动态规划计算编辑距离
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # 删除
                        dp[i][j-1] + 1,      # 插入
                        dp[i-1][j-1] + 1     # 替换
                    )
        
        return dp[m][n]
    
    @classmethod
    def normalized_levenshtein_similarity(cls, text1: str, text2: str) -> float:
        """标准化的编辑距离相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            float: 相似度分数 (0-1)
        """
        if not text1 and not text2:
            return 1.0
        
        distance = cls.levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        return 1 - (distance / max_len) if max_len > 0 else 0.0


class TextHasher:
    """文本哈希器
    
    用于生成文本的唯一标识符
    """
    
    @staticmethod
    def md5_hash(text: str) -> str:
        """生成MD5哈希
        
        Args:
            text: 输入文本
            
        Returns:
            str: MD5哈希值
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def sha256_hash(text: str) -> str:
        """生成SHA256哈希
        
        Args:
            text: 输入文本
            
        Returns:
            str: SHA256哈希值
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def content_hash(text: str, algorithm: str = 'md5') -> str:
        """生成内容哈希
        
        Args:
            text: 输入文本
            algorithm: 哈希算法 ('md5', 'sha256')
            
        Returns:
            str: 哈希值
        """
        if algorithm == 'md5':
            return TextHasher.md5_hash(text)
        elif algorithm == 'sha256':
            return TextHasher.sha256_hash(text)
        else:
            raise ValueError(f"不支持的哈希算法: {algorithm}")


# 便捷函数
def clean_text(text: str, **kwargs) -> str:
    """清理文本的便捷函数
    
    Args:
        text: 输入文本
        **kwargs: 清理选项
        
    Returns:
        str: 清理后的文本
    """
    cleaner = TextCleaner()
    return cleaner.clean_text(text, **kwargs)


def analyze_text(text: str) -> TextMetrics:
    """分析文本的便捷函数
    
    Args:
        text: 输入文本
        
    Returns:
        TextMetrics: 分析结果
    """
    analyzer = TextAnalyzer()
    return analyzer.get_text_metrics(text)


def extract_keywords(text: str, top_k: int = 10) -> List[Tuple[str, int]]:
    """提取关键词的便捷函数
    
    Args:
        text: 输入文本
        top_k: 返回前k个关键词
        
    Returns:
        List[Tuple[str, int]]: 关键词列表
    """
    analyzer = TextAnalyzer()
    return analyzer.extract_keywords(text, top_k)


def calculate_similarity(text1: str, text2: str, method: str = 'cosine') -> float:
    """计算文本相似度的便捷函数
    
    Args:
        text1: 文本1
        text2: 文本2
        method: 相似度方法 ('jaccard', 'cosine', 'levenshtein')
        
    Returns:
        float: 相似度分数
    """
    if method == 'jaccard':
        return TextSimilarity.jaccard_similarity(text1, text2)
    elif method == 'cosine':
        return TextSimilarity.cosine_similarity(text1, text2)
    elif method == 'levenshtein':
        return TextSimilarity.normalized_levenshtein_similarity(text1, text2)
    else:
        raise ValueError(f"不支持的相似度方法: {method}")


def generate_text_hash(text: str, algorithm: str = 'md5') -> str:
    """生成文本哈希的便捷函数
    
    Args:
        text: 输入文本
        algorithm: 哈希算法
        
    Returns:
        str: 哈希值
    """
    return TextHasher.content_hash(text, algorithm)


# 预定义实例
default_cleaner = TextCleaner()
default_analyzer = TextAnalyzer()