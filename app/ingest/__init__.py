"""文档摄取模块

提供文档解析、预处理、分段和处理管道功能。
支持多种文档格式的处理和灵活的分段策略。
"""

from .parser import (
    DocumentParser,
    ParseResult,
    Parser,
    TxtParser,
    MarkdownParser,
    HtmlParser,
    PdfParser
)

from .preprocessor import (
    TextPreprocessor,
    PreprocessConfig
)

from .splitter import (
    SplitMode,
    ChunkMetadata,
    TextChunk,
    SplitterConfig,
    BaseSplitter,
    FlatSplitter,
    ParentChildSplitter,
    TextSplitter,
    create_splitter
)

from .pipeline import (
    ProcessingResult,
    PipelineConfig,
    DocumentProcessor,
    PipelineBuilder,
    create_pipeline,
    create_pipeline_builder,
    process_single_file,
    process_multiple_files,
    process_directory_files
)

# 版本信息
__version__ = "1.0.0"

# 导出的公共接口
__all__ = [
    # Parser相关
    "Parser",
    "TxtParser",
    "MarkdownParser", 
    "HtmlParser",
    "PdfParser",
    "DocumentParser",
    "ParseResult",
    
    # Preprocessor相关
    "TextPreprocessor",
    "PreprocessConfig",
    
    # Splitter相关
    "SplitMode",
    "ChunkMetadata",
    "TextChunk",
    "SplitterConfig",
    "BaseSplitter",
    "FlatSplitter",
    "ParentChildSplitter",
    "TextSplitter",
    "create_splitter",
    
    # Pipeline相关
    "ProcessingResult",
    "PipelineConfig",
    "DocumentProcessor",
    "PipelineBuilder",
    "create_pipeline",
    "create_pipeline_builder",
    "process_single_file",
    "process_multiple_files",
    "process_directory_files",
]

# 便捷的工厂函数
def create_default_pipeline(chunk_size: int = 1000, 
                           chunk_overlap: int = 200,
                           split_mode: SplitMode = SplitMode.FLAT) -> DocumentProcessor:
    """创建默认的文档处理管道
    
    Args:
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        split_mode: 分段模式
        
    Returns:
        DocumentProcessor: 文档处理器
    """
    return create_pipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        split_mode=split_mode
    )


def quick_process(file_path: str, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200) -> ProcessingResult:
    """快速处理单个文档的便捷函数
    
    Args:
        file_path: 文档路径
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        
    Returns:
        ProcessingResult: 处理结果
    """
    pipeline = create_default_pipeline(chunk_size, chunk_overlap)
    return pipeline.process_file(file_path)