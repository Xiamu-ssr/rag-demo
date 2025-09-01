#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试解析与分段模块功能

验证文档解析器、预处理器、分段器和处理管道的基本功能。
"""

import os
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ingest.parser import DocumentParser
from app.ingest.preprocessor import TextPreprocessor
from app.ingest.splitter import TextSplitter, SplitterConfig, SplitMode
from app.ingest.pipeline import DocumentProcessor, create_pipeline_builder, process_single_file
from app.utils.token_counter import count_tokens, get_token_stats
from app.utils.text_utils import clean_text, analyze_text, calculate_similarity


def test_parser():
    """测试文档解析器"""
    print("\n=== 测试文档解析器 ===")
    
    parser = DocumentParser()
    
    # 测试TXT解析
    txt_content = "This is a test document.\nContains mixed Chinese and English content.\nUsed for testing parsing functionality."
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(txt_content)
        txt_file = f.name
    
    try:
        with open(txt_file, 'rb') as f:
            content = f.read()
        result = parser.parse(content, txt_file)
        print(f"✓ TXT file parsed: {len(result.text)} characters")
        print(f"  Title: {result.meta.get('title', 'N/A')}")
        print(f"  Format: {result.meta.get('format', 'N/A')}")
    finally:
        os.unlink(txt_file)
    
    # 测试Markdown解析
    md_content = "# Title\n\nThis is **bold** text and *italic* text.\n\n- List item 1\n- List item 2"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(md_content)
        md_file = f.name
    
    try:
        with open(md_file, 'rb') as f:
            content = f.read()
        result = parser.parse(content, md_file)
        print(f"✓ Markdown file parsed: {len(result.text)} characters")
        print(f"  Title: {result.meta.get('title', 'N/A')}")
        print(f"  Format: {result.meta.get('format', 'N/A')}")
    finally:
        os.unlink(md_file)
    
    # 测试HTML解析
    html_content = "<html><body><h1>Title</h1><p>This is an <strong>HTML</strong> document.</p></body></html>"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        f.write(html_content)
        html_file = f.name
    
    try:
        with open(html_file, 'rb') as f:
            content = f.read()
        result = parser.parse(content, html_file)
        print(f"✓ HTML file parsed: {len(result.text)} characters")
        print(f"  Title: {result.meta.get('title', 'N/A')}")
        print(f"  Format: {result.meta.get('format', 'N/A')}")
    finally:
        os.unlink(html_file)


def test_preprocessor():
    """测试文本预处理器"""
    print("\n=== 测试文本预处理器 ===")
    
    preprocessor = TextPreprocessor()
    
    # 测试基本预处理
    text = "This is a test document.\n\n\nContains full-width chars: １２３ＡＢＣ.\nAlso has URL: https://example.com and email: test@example.com"
    
    processed = preprocessor.preprocess(text)
    print(f"Original text: {repr(text)}")
    print(f"Processed: {repr(processed)}")
    
    # 验证全角转半角
    assert "123ABC" in processed
    
    print("✓ Text preprocessing test passed")


def test_splitter():
    """测试文本分段器"""
    print("\n=== 测试文本分段器 ===")
    
    # 创建测试文本
    text = "This is the first paragraph. " * 50 + "\n\n" + "This is the second paragraph. " * 50 + "\n\n" + "This is the third paragraph. " * 50
    
    # Test flat splitting
    config = SplitterConfig(chunk_size=200, chunk_overlap=50, split_mode=SplitMode.FLAT)
    splitter = TextSplitter(config)
    chunks = splitter.split_text(text)
    
    print(f"Flat splitting: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"  Chunk {i+1}: {len(chunk.content)} chars - '{chunk.content[:50]}...'")
    
    assert len(chunks) > 1
    # 检查大部分分段都在合理范围内（允许一些超出，因为要保持句子完整性）
    reasonable_chunks = [chunk for chunk in chunks if len(chunk.content) <= 300]
    assert len(reasonable_chunks) >= len(chunks) * 0.8  # 至少80%的分段在合理范围内
    print("✓ Flat splitting test passed")
    
    # Test parent-child splitting
    config = SplitterConfig(chunk_size=200, chunk_overlap=50, split_mode=SplitMode.PARENT_CHILD)
    splitter = TextSplitter(config)
    chunks = splitter.split_text(text)
    
    print(f"Parent-child splitting: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"  Chunk {i+1}: {len(chunk.content)} chars - '{chunk.content[:50]}...'")
        if chunk.metadata.parent_id:
            print(f"    Parent ID: {chunk.metadata.parent_id}")
    
    assert len(chunks) > 1
    print("✓ Parent-child splitting test passed")


def test_pipeline():
    """测试处理管道"""
    print("\n=== 测试处理管道 ===")
    
    # 创建测试文档
    content = "# Test Document\n\nThis is a test document with **important content**.\n\n" + "Test paragraph. " * 100
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(content)
        test_file = f.name
    
    try:
        # 使用管道构建器创建管道
        builder = create_pipeline_builder()
        splitter_config = {'chunk_size': 300, 'chunk_overlap': 50}
        pipeline = builder.with_splitter_config(splitter_config).build()
        result = pipeline.process_file(test_file)
        
        print(f"Processing result status: {result.success}")
        print(f"Chunk count: {len(result.chunks)}")
        print(f"First chunk: {result.chunks[0].content[:50]}...")
        print(f"Processing time: {result.processing_time:.3f}s")
        
        assert result.success
        assert len(result.chunks) > 0
        assert "Test Document" in result.chunks[0].content
        print("✓ Processing pipeline test passed")
        
        # 测试单文件处理函数
        config = {'splitter_config': {'chunk_size': 200}}
        quick_result = process_single_file(test_file, config)
        print(f"Single file processing chunk count: {len(quick_result.chunks)}")
        assert quick_result.success
        print("✓ Single file processing test passed")
        
    finally:
        os.unlink(test_file)


def test_token_counter():
    """测试token计数工具"""
    print("\n=== 测试Token计数工具 ===")
    
    text = "This is a test text with mixed Chinese and English content. Used for testing token counting functionality."
    
    # 测试基本计数
    token_count = count_tokens(text)
    print(f"Text: {text}")
    print(f"Token count: {token_count}")
    
    # 测试详细统计
    stats = get_token_stats(text)
    print(f"Detailed stats:")
    print(f"  Character count: {stats.char_count}")
    print(f"  Word count: {stats.word_count}")
    print(f"  Chinese char count: {stats.chinese_char_count}")
    print(f"  English word count: {stats.english_word_count}")
    print(f"  Chars per token: {stats.chars_per_token:.2f}")
    
    assert token_count > 0
    assert stats.char_count == len(text)
    assert stats.english_word_count > 0
    print("✓ Token counting test passed")


def test_text_utils():
    """测试文本处理工具"""
    print("\n=== 测试文本处理工具 ===")
    
    # 测试文本清理
    dirty_text = "This is a test document. Contains URL: https://example.com and <strong>HTML tags</strong>."
    clean = clean_text(dirty_text)
    print(f"Original text: {dirty_text}")
    print(f"Cleaned: {clean}")
    
    assert "https://example.com" not in clean
    assert "<strong>" not in clean
    print("✓ Text cleaning test passed")
    
    # 测试文本分析
    analysis_text = "This is a test document. Contains Chinese and English content. Used for analysis testing."
    metrics = analyze_text(analysis_text)
    print(f"Text analysis results:")
    print(f"  Length: {metrics.length}")
    print(f"  Word count: {metrics.word_count}")
    print(f"  Chinese char count: {metrics.chinese_char_count}")
    print(f"  English word count: {metrics.english_word_count}")
    print(f"  Chinese ratio: {metrics.chinese_ratio:.2f}")
    
    assert metrics.length > 0
    assert metrics.english_word_count > 0
    print("✓ Text analysis test passed")
    
    # 测试相似度计算
    text1 = "This is the first test document"
    text2 = "This is the second test document"
    similarity = calculate_similarity(text1, text2, method='cosine')
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Cosine similarity: {similarity:.3f}")
    
    assert 0 <= similarity <= 1
    assert similarity > 0.5  # Should have high similarity
    print("✓ Similarity calculation test passed")


def test_integration():
    """集成测试"""
    print("\n=== 集成测试 ===")
    
    # 创建一个复杂的测试文档
    complex_content = """
# RAG System Test Document

## Introduction
This is a document for testing RAG (Retrieval-Augmented Generation) system.

## Main Features

### Document Parsing
- Support multiple formats: PDF, TXT, HTML, Markdown
- Extract text content and metadata
- Handle mixed Chinese and English content

### Text Preprocessing
- Unicode normalization
- Full-width to half-width conversion
- Whitespace normalization
- URL and email removal

### Text Splitting
- Flat splitting mode
- Parent-child splitting mode
- Configurable chunk size and overlap

### Token Counting
- Support multiple tokenizers
- Accurate token statistics
- Cost estimation functionality

## Test Cases

Here are some test contents for verifying system functionality. Contains sufficient text length for testing splitting functionality.
""" + "Repeated content for testing. " * 50 + """

## Conclusion

Through this test document, we can verify that the core functionality of the RAG system works properly.
"""
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(complex_content)
        test_file = f.name
    
    try:
        # 完整的处理流程
        print("Starting complete processing workflow...")
        
        # 1. 解析文档
        parser = DocumentParser()
        with open(test_file, 'rb') as f:
            content = f.read()
        parsed = parser.parse(content, test_file)
        print(f"1. Document parsing completed, content length: {len(parsed.text)}")
        
        # 2. 预处理
        preprocessor = TextPreprocessor()
        preprocessed = preprocessor.preprocess(parsed.text)
        print(f"2. Text preprocessing completed, processed length: {len(preprocessed)}")
        
        # 3. 分段
        config = SplitterConfig(chunk_size=500, chunk_overlap=100, split_mode=SplitMode.PARENT_CHILD)
        splitter = TextSplitter(config)
        chunks = splitter.split_text(preprocessed)
        print(f"3. Text splitting completed, chunk count: {len(chunks)}")
        
        # 4. Token统计
        total_tokens = sum(count_tokens(chunk.content) for chunk in chunks)
        print(f"4. Token counting completed, total tokens: {total_tokens}")
        
        # 5. 使用管道处理（对比）
        builder = create_pipeline_builder()
        splitter_config = {'chunk_size': 500, 'chunk_overlap': 100}
        pipeline = builder.with_splitter_config(splitter_config).build()
        pipeline_result = pipeline.process_file(test_file)
        print(f"5. Pipeline processing completed, chunk count: {len(pipeline_result.chunks)}")
        
        # 验证结果一致性
        print("\n=== 验证处理结果 ===")
        print(f"Manual chunks: {len(chunks)}, Pipeline chunks: {len(pipeline_result.chunks)}")
        print(f"Manual tokens: {total_tokens}, Pipeline tokens: {pipeline_result.total_tokens}")
        
        # 基本验证（由于配置不同，分段数量可能不同，但都应该有合理的结果）
        assert len(chunks) > 0
        assert len(pipeline_result.chunks) > 0
        assert total_tokens > 0
        assert pipeline_result.total_tokens > 0
        assert pipeline_result.success
        
        print("✓ Integration test passed - All components work together correctly!")
        
    finally:
        os.unlink(test_file)


def main():
    """运行所有测试"""
    print("Starting tests for parsing and splitting module functionality...")
    
    try:
        test_parser()
        test_preprocessor()
        test_splitter()
        test_pipeline()
        test_token_counter()
        test_text_utils()
        test_integration()
        
        print("\n" + "="*50)
        print("🎉 All tests passed! Parsing and splitting module functionality is working properly.")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()