"""文档解析器模块

支持PDF、TXT、HTML、Markdown格式的文档解析，
将文件内容转换为纯文本并提取元数据。
"""

import hashlib
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

from pydantic import BaseModel


class ParseResult(BaseModel):
    """解析结果模型"""
    text: str
    meta: Dict[str, Any]  # pages, mime, sha256, title, etc.


class Parser(ABC):
    """文档解析器抽象基类"""
    
    @abstractmethod
    def parse(self, file: bytes, filename: str) -> ParseResult:
        """解析文件内容
        
        Args:
            file: 文件字节内容
            filename: 文件名
            
        Returns:
            ParseResult: 解析结果，包含文本和元数据
        """
        pass
    
    @abstractmethod
    def supports(self, filename: str) -> bool:
        """检查是否支持该文件格式
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否支持
        """
        pass
    
    def _get_file_hash(self, content: bytes) -> str:
        """计算文件SHA256哈希值"""
        return hashlib.sha256(content).hexdigest()
    
    def _get_mime_type(self, filename: str) -> str:
        """获取文件MIME类型"""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"
    
    def _extract_title(self, filename: str, text: str) -> str:
        """提取文档标题"""
        # 优先使用文件名作为标题
        title = Path(filename).stem
        
        # 尝试从文本内容提取标题
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # 如果第一行不太长且不为空，可能是标题
            if first_line and len(first_line) < 100:
                title = first_line
        
        return title


class TxtParser(Parser):
    """TXT文件解析器"""
    
    def supports(self, filename: str) -> bool:
        return filename.lower().endswith('.txt')
    
    def parse(self, file: bytes, filename: str) -> ParseResult:
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            text = None
            
            for encoding in encodings:
                try:
                    text = file.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise ValueError("无法解码文本文件")
            
            meta = {
                'mime': self._get_mime_type(filename),
                'sha256': self._get_file_hash(file),
                'size': len(file),
                'title': self._extract_title(filename, text),
                'encoding': encoding
            }
            
            return ParseResult(text=text, meta=meta)
            
        except Exception as e:
            raise ValueError(f"TXT文件解析失败: {str(e)}")


class MarkdownParser(Parser):
    """Markdown文件解析器"""
    
    def supports(self, filename: str) -> bool:
        return filename.lower().endswith(('.md', '.markdown'))
    
    def parse(self, file: bytes, filename: str) -> ParseResult:
        try:
            # 解码文本
            encodings = ['utf-8', 'gbk', 'gb2312']
            text = None
            
            for encoding in encodings:
                try:
                    text = file.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise ValueError("无法解码Markdown文件")
            
            # 提取标题（第一个# 标题）
            title = self._extract_title(filename, text)
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('# '):
                    title = line[2:].strip()
                    break
            
            meta = {
                'mime': self._get_mime_type(filename),
                'sha256': self._get_file_hash(file),
                'size': len(file),
                'title': title,
                'format': 'markdown'
            }
            
            return ParseResult(text=text, meta=meta)
            
        except Exception as e:
            raise ValueError(f"Markdown文件解析失败: {str(e)}")


class HtmlParser(Parser):
    """HTML文件解析器"""
    
    def supports(self, filename: str) -> bool:
        return filename.lower().endswith(('.html', '.htm'))
    
    def parse(self, file: bytes, filename: str) -> ParseResult:
        try:
            # 解码HTML
            encodings = ['utf-8', 'gbk', 'gb2312']
            html_content = None
            
            for encoding in encodings:
                try:
                    html_content = file.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if html_content is None:
                raise ValueError("无法解码HTML文件")
            
            # 简单的HTML标签清理（可以后续使用BeautifulSoup优化）
            import re
            
            # 提取title标签内容
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else self._extract_title(filename, html_content)
            
            # 移除HTML标签，保留文本内容
            # 先处理换行相关的标签
            html_content = re.sub(r'<(br|p|div|h[1-6])[^>]*>', '\n', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'</(p|div|h[1-6])>', '\n', html_content, flags=re.IGNORECASE)
            
            # 移除所有HTML标签
            text = re.sub(r'<[^>]+>', '', html_content)
            
            # 解码HTML实体
            import html
            text = html.unescape(text)
            
            # 清理多余的空白字符
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = text.strip()
            
            meta = {
                'mime': self._get_mime_type(filename),
                'sha256': self._get_file_hash(file),
                'size': len(file),
                'title': title,
                'format': 'html'
            }
            
            return ParseResult(text=text, meta=meta)
            
        except Exception as e:
            raise ValueError(f"HTML文件解析失败: {str(e)}")


class PdfParser(Parser):
    """PDF文件解析器
    
    注意：需要安装PyPDF2或pdfplumber库
    这里提供基础实现，可根据需要选择更强大的PDF解析库
    """
    
    def supports(self, filename: str) -> bool:
        return filename.lower().endswith('.pdf')
    
    def parse(self, file: bytes, filename: str) -> ParseResult:
        try:
            # 尝试使用PyPDF2解析PDF
            try:
                import PyPDF2
                from io import BytesIO
                
                pdf_reader = PyPDF2.PdfReader(BytesIO(file))
                text_parts = []
                
                for page in pdf_reader.pages:
                    text_parts.append(page.extract_text())
                
                text = '\n'.join(text_parts)
                page_count = len(pdf_reader.pages)
                
                # 尝试获取PDF元数据
                pdf_info = pdf_reader.metadata or {}
                title = pdf_info.get('/Title', self._extract_title(filename, text))
                author = pdf_info.get('/Author', '')
                
            except ImportError:
                # 如果没有PyPDF2，尝试使用pdfplumber
                try:
                    import pdfplumber
                    from io import BytesIO
                    
                    with pdfplumber.open(BytesIO(file)) as pdf:
                        text_parts = []
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(page_text)
                        
                        text = '\n'.join(text_parts)
                        page_count = len(pdf.pages)
                        title = self._extract_title(filename, text)
                        author = ''
                        
                except ImportError:
                    raise ValueError("需要安装PyPDF2或pdfplumber库来解析PDF文件")
            
            meta = {
                'mime': self._get_mime_type(filename),
                'sha256': self._get_file_hash(file),
                'size': len(file),
                'title': title,
                'author': author,
                'pages': page_count,
                'format': 'pdf'
            }
            
            return ParseResult(text=text, meta=meta)
            
        except Exception as e:
            raise ValueError(f"PDF文件解析失败: {str(e)}")


class DocumentParser:
    """文档解析器管理类"""
    
    def __init__(self):
        self.parsers = [
            TxtParser(),
            MarkdownParser(),
            HtmlParser(),
            PdfParser()
        ]
    
    def parse(self, file: bytes, filename: str) -> ParseResult:
        """解析文档
        
        Args:
            file: 文件字节内容
            filename: 文件名
            
        Returns:
            ParseResult: 解析结果
            
        Raises:
            ValueError: 不支持的文件格式或解析失败
        """
        for parser in self.parsers:
            if parser.supports(filename):
                return parser.parse(file, filename)
        
        raise ValueError(f"不支持的文件格式: {filename}")
    
    def supports(self, filename: str) -> bool:
        """检查是否支持该文件格式"""
        return any(parser.supports(filename) for parser in self.parsers)
    
    def get_supported_extensions(self) -> list[str]:
        """获取支持的文件扩展名列表"""
        extensions = []
        test_files = [
            'test.txt', 'test.md', 'test.markdown', 
            'test.html', 'test.htm', 'test.pdf'
        ]
        
        for filename in test_files:
            if self.supports(filename):
                ext = Path(filename).suffix
                if ext not in extensions:
                    extensions.append(ext)
        
        return extensions