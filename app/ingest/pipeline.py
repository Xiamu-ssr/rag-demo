"""文档处理管道模块

整合文档解析、预处理、分段等流程，提供统一的文档处理接口。
支持批量处理和流式处理模式。
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .parser import DocumentParser
from .preprocessor import TextPreprocessor, PreprocessConfig, create_preprocessor
from .splitter import TextSplitter, SplitterConfig, TextChunk, create_splitter

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool                           # 是否成功
    file_path: str                         # 文件路径
    chunks: List[TextChunk] = field(default_factory=list)  # 分段结果
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    error: Optional[str] = None            # 错误信息
    processing_time: float = 0.0           # 处理时间（秒）
    
    @property
    def chunk_count(self) -> int:
        """分段数量"""
        return len(self.chunks)
    
    @property
    def total_chars(self) -> int:
        """总字符数"""
        return sum(chunk.metadata.char_count for chunk in self.chunks)
    
    @property
    def total_tokens(self) -> int:
        """总token数"""
        return sum(chunk.metadata.token_count for chunk in self.chunks)


@dataclass
class PipelineConfig:
    """管道配置"""
    # 解析器配置
    parser_config: Optional[Dict[str, Any]] = None
    
    # 预处理器配置
    preprocessor_config: Optional[Dict[str, Any]] = None
    
    # 分段器配置
    splitter_config: Optional[Dict[str, Any]] = None
    
    # 处理配置
    max_workers: int = 4                   # 最大并发数
    batch_size: int = 10                   # 批处理大小
    enable_async: bool = True              # 启用异步处理
    
    # 错误处理
    skip_errors: bool = True               # 跳过错误文件
    max_retries: int = 3                   # 最大重试次数
    
    # 输出配置
    save_intermediate: bool = False        # 保存中间结果
    output_format: str = "json"            # 输出格式


class DocumentProcessor:
    """文档处理器
    
    整合解析、预处理、分段等步骤的核心处理类
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # 初始化各个组件
        self.parser = DocumentParser()
        self.preprocessor = create_preprocessor(self.config.preprocessor_config)
        self.splitter = create_splitter(self.config.splitter_config)
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # 统计信息
        self.stats = {
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'total_chars': 0,
            'total_tokens': 0,
            'processing_time': 0.0
        }
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """处理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            ProcessingResult: 处理结果
        """
        import time
        start_time = time.time()
        
        file_path = str(file_path)
        
        try:
            # 1. 解析文档
            logger.info(f"开始解析文件: {file_path}")
            # 读取文件内容（以二进制模式读取）
            with open(file_path, 'rb') as f:
                content = f.read()
            parse_result = self.parser.parse(content, file_path)
            
            # ParseResult总是成功的，如果失败会抛出异常
            # 2. 预处理文本
            logger.info(f"开始预处理文本: {file_path}")
            preprocessed_text = self.preprocessor.preprocess(parse_result.text)
            
            if not preprocessed_text or not preprocessed_text.strip():
                return ProcessingResult(
                    success=False,
                    file_path=file_path,
                    error="预处理后文本为空",
                    processing_time=time.time() - start_time
                )
            
            # 3. 文本分段
            logger.info(f"开始文本分段: {file_path}")
            chunks = self.splitter.split_text(preprocessed_text, parse_result.meta)
            
            if not chunks:
                return ProcessingResult(
                    success=False,
                    file_path=file_path,
                    error="分段结果为空",
                    processing_time=time.time() - start_time
                )
            
            # 4. 构建结果
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                success=True,
                file_path=file_path,
                chunks=chunks,
                metadata={
                    **parse_result.meta,
                    'preprocessing_stats': self.preprocessor.get_text_stats(preprocessed_text),
                    'splitting_stats': self.splitter.get_statistics(chunks)
                },
                processing_time=processing_time
            )
            
            # 更新统计信息
            self._update_stats(result)
            
            logger.info(f"文件处理完成: {file_path}, 耗时: {processing_time:.2f}s, 分段数: {len(chunks)}")
            
            return result
            
        except Exception as e:
            error_msg = f"处理文件时发生错误: {str(e)}"
            logger.error(f"{error_msg}, 文件: {file_path}")
            
            return ProcessingResult(
                success=False,
                file_path=file_path,
                error=error_msg,
                processing_time=time.time() - start_time
            )
    
    def process_files(self, file_paths: List[Union[str, Path]]) -> List[ProcessingResult]:
        """批量处理文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        results = []
        
        if self.config.enable_async and len(file_paths) > 1:
            # 异步并发处理
            logger.info(f"开始并发处理 {len(file_paths)} 个文件")
            
            futures = []
            for file_path in file_paths:
                future = self.executor.submit(self.process_file, file_path)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if not result.success and not self.config.skip_errors:
                        logger.error(f"处理失败，停止批量处理: {result.error}")
                        break
                        
                except Exception as e:
                    error_result = ProcessingResult(
                        success=False,
                        file_path="unknown",
                        error=f"获取处理结果时发生错误: {str(e)}"
                    )
                    results.append(error_result)
                    
                    if not self.config.skip_errors:
                        break
        else:
            # 同步顺序处理
            logger.info(f"开始顺序处理 {len(file_paths)} 个文件")
            
            for file_path in file_paths:
                result = self.process_file(file_path)
                results.append(result)
                
                if not result.success and not self.config.skip_errors:
                    logger.error(f"处理失败，停止批量处理: {result.error}")
                    break
        
        return results
    
    def process_directory(self, 
                         directory: Union[str, Path], 
                         pattern: str = "*",
                         recursive: bool = True) -> List[ProcessingResult]:
        """处理目录中的文件
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            recursive: 是否递归处理子目录
            
        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            logger.error(f"目录不存在或不是有效目录: {directory}")
            return []
        
        # 查找文件
        if recursive:
            file_paths = list(directory.rglob(pattern))
        else:
            file_paths = list(directory.glob(pattern))
        
        # 过滤出文件（排除目录）
        file_paths = [p for p in file_paths if p.is_file()]
        
        # 过滤支持的文件类型
        supported_extensions = {'.txt', '.md', '.html', '.htm', '.pdf'}
        file_paths = [p for p in file_paths if p.suffix.lower() in supported_extensions]
        
        logger.info(f"在目录 {directory} 中找到 {len(file_paths)} 个支持的文件")
        
        if not file_paths:
            return []
        
        return self.process_files(file_paths)
    
    async def process_file_async(self, file_path: Union[str, Path]) -> ProcessingResult:
        """异步处理单个文件"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.process_file, file_path)
    
    async def process_files_async(self, file_paths: List[Union[str, Path]]) -> AsyncGenerator[ProcessingResult, None]:
        """异步流式处理文件
        
        Args:
            file_paths: 文件路径列表
            
        Yields:
            ProcessingResult: 处理结果
        """
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process_file_async(file_path)
        
        # 创建任务
        tasks = [process_with_semaphore(file_path) for file_path in file_paths]
        
        # 逐个返回完成的结果
        for coro in asyncio.as_completed(tasks):
            result = await coro
            yield result
    
    def _update_stats(self, result: ProcessingResult):
        """更新统计信息"""
        if result.success:
            self.stats['processed_files'] += 1
            self.stats['total_chunks'] += result.chunk_count
            self.stats['total_chars'] += result.total_chars
            self.stats['total_tokens'] += result.total_tokens
        else:
            self.stats['failed_files'] += 1
        
        self.stats['processing_time'] += result.processing_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        total_files = self.stats['processed_files'] + self.stats['failed_files']
        
        return {
            **self.stats,
            'total_files': total_files,
            'success_rate': self.stats['processed_files'] / total_files if total_files > 0 else 0,
            'avg_processing_time': self.stats['processing_time'] / total_files if total_files > 0 else 0,
            'avg_chunks_per_file': self.stats['total_chunks'] / self.stats['processed_files'] if self.stats['processed_files'] > 0 else 0
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'total_chars': 0,
            'total_tokens': 0,
            'processing_time': 0.0
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)


class PipelineBuilder:
    """管道构建器
    
    提供流式API来配置和构建处理管道
    """
    
    def __init__(self):
        self.config = PipelineConfig()
    
    def with_parser_config(self, config: Dict[str, Any]) -> 'PipelineBuilder':
        """配置解析器"""
        self.config.parser_config = config
        return self
    
    def with_preprocessor_config(self, config: Dict[str, Any]) -> 'PipelineBuilder':
        """配置预处理器"""
        self.config.preprocessor_config = config
        return self
    
    def with_splitter_config(self, config: Dict[str, Any]) -> 'PipelineBuilder':
        """配置分段器"""
        self.config.splitter_config = config
        return self
    
    def with_concurrency(self, max_workers: int) -> 'PipelineBuilder':
        """配置并发数"""
        self.config.max_workers = max_workers
        return self
    
    def with_batch_size(self, batch_size: int) -> 'PipelineBuilder':
        """配置批处理大小"""
        self.config.batch_size = batch_size
        return self
    
    def enable_async_processing(self, enable: bool = True) -> 'PipelineBuilder':
        """启用/禁用异步处理"""
        self.config.enable_async = enable
        return self
    
    def skip_errors(self, skip: bool = True) -> 'PipelineBuilder':
        """配置错误处理策略"""
        self.config.skip_errors = skip
        return self
    
    def build(self) -> DocumentProcessor:
        """构建处理器"""
        return DocumentProcessor(self.config)


def create_pipeline(config_dict: Optional[Dict[str, Any]] = None) -> DocumentProcessor:
    """创建文档处理管道
    
    Args:
        config_dict: 配置字典
        
    Returns:
        DocumentProcessor: 文档处理器实例
    """
    if config_dict:
        config = PipelineConfig(**config_dict)
    else:
        config = PipelineConfig()
    
    return DocumentProcessor(config)


def create_pipeline_builder() -> PipelineBuilder:
    """创建管道构建器
    
    Returns:
        PipelineBuilder: 管道构建器实例
    """
    return PipelineBuilder()


# 便捷函数
def process_single_file(file_path: Union[str, Path], 
                       config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
    """处理单个文件的便捷函数
    
    Args:
        file_path: 文件路径
        config: 配置字典
        
    Returns:
        ProcessingResult: 处理结果
    """
    with create_pipeline(config) as processor:
        return processor.process_file(file_path)


def process_multiple_files(file_paths: List[Union[str, Path]], 
                          config: Optional[Dict[str, Any]] = None) -> List[ProcessingResult]:
    """处理多个文件的便捷函数
    
    Args:
        file_paths: 文件路径列表
        config: 配置字典
        
    Returns:
        List[ProcessingResult]: 处理结果列表
    """
    with create_pipeline(config) as processor:
        return processor.process_files(file_paths)


def process_directory_files(directory: Union[str, Path],
                           pattern: str = "*",
                           recursive: bool = True,
                           config: Optional[Dict[str, Any]] = None) -> List[ProcessingResult]:
    """处理目录中文件的便捷函数
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        recursive: 是否递归处理
        config: 配置字典
        
    Returns:
        List[ProcessingResult]: 处理结果列表
    """
    with create_pipeline(config) as processor:
        return processor.process_directory(directory, pattern, recursive)