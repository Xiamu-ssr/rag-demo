#!/usr/bin/env python3
"""RAG系统索引和检索功能测试脚本

测试内容：
1. 嵌入器功能测试
2. FAISS存储功能测试
3. 索引构建功能测试
4. 向量检索功能测试
5. MMR去冗余测试
6. 多库融合测试
7. 完整流程性能测试
"""

import time
import numpy as np
from typing import List, Dict, Any
import logging
from app.db.models import Collection

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_embedding_model():
    """设置嵌入模型"""
    try:
        from app.llm import get_model_manager, ModelType
        
        # 获取模型管理器
        model_manager = get_model_manager()
        
        # 切换到嵌入模型（这会设置为当前模型）
        provider = model_manager.switch_embedding_model("bge-m3")
        logger.info(f"嵌入模型已切换: {provider.config.model_id}")
        return True
    except Exception as e:
        logger.error(f"嵌入模型切换失败: {e}")
        return False


def test_embedder():
    """测试嵌入器功能"""
    logger.info("开始测试嵌入器功能...")
    
    try:
        from app.llm import get_model_manager, ModelType
        
        # 获取模型管理器
        model_manager = get_model_manager()
        
        # 获取当前嵌入模型
        provider = model_manager.get_current_embedding_model()
        if not provider:
            logger.error("没有可用的嵌入提供商")
            assert False, "没有可用的嵌入提供商"
        
        # 测试单文本嵌入
        test_text = "这是一个测试文本"
        embedding = provider.embed_query(test_text)
        
        assert len(embedding) > 0, "嵌入向量长度应大于0"
        assert isinstance(embedding, (list, np.ndarray)), "嵌入结果应为列表或数组"
        
        # 测试批量嵌入
        test_texts = ["文本1", "文本2", "文本3"]
        embeddings = provider.embed_documents(test_texts)
        
        assert len(embeddings) == len(test_texts), "批量嵌入结果数量应匹配输入"
        
        logger.info("✓ 嵌入器功能测试通过")
        
    except Exception as e:
        logger.error(f"✗ 嵌入器功能测试失败: {e}")
        assert False, f"嵌入器功能测试失败: {e}"

def test_faiss_store():
    """测试FAISS存储功能"""
    logger.info("开始测试FAISS存储功能...")
    
    try:
        from app.storage import VectorStoreFactory
        
        # 创建FAISS存储实例
        store = VectorStoreFactory.create(
            "faiss",
            {
                "storage_path": "/tmp/test_faiss",
                "dimension": 128
            }
        )
        
        # 生成测试向量
        test_vectors = np.random.random((10, 128)).astype(np.float32)
        test_ids = [f"doc_{i}" for i in range(10)]
        
        # 准备向量记录
        from app.storage.base import VectorRecord
        records = [
            VectorRecord(id=test_ids[i], vector=test_vectors[i], metadata={})
            for i in range(10)
        ]
        
        # 创建集合
        store.create_collection("test_collection", 128)
        
        # 测试添加向量
        store.insert_vectors("test_collection", records)
        
        # 测试搜索
        query_vector = test_vectors[0]  # 使用第一个向量作为查询
        results = store.search_vectors("test_collection", query_vector, top_k=5)
        
        assert len(results) == 5, "TopK结果数量不匹配"
        assert results[0].id == "doc_0", "最相似结果不匹配"
        
        # 测试删除向量
        store.delete_vectors("test_collection", ["doc_0", "doc_1"])
        
        logger.info("✓ FAISS存储功能测试通过")
        
    except Exception as e:
        logger.error(f"✗ FAISS存储功能测试失败: {e}")
        assert False, f"FAISS存储功能测试失败: {e}"

def test_index_builder():
    """测试索引构建功能"""
    logger.info("开始测试索引构建功能...")
    
    try:
        from app.indexer import IndexBuilder
        from app.llm import get_model_manager, ModelType
        from app.core.database import get_session
        from app.db.models import Collection, Document, Chunk
        import uuid
        from datetime import datetime
        
        # 获取模型管理器
        model_manager = get_model_manager()
        
        # 获取嵌入模型提供商
        embedder = model_manager.get_provider(ModelType.EMBEDDING)
        
        session = next(get_session())
        try:
            # 检查集合是否存在，不存在则创建
            collection = session.query(Collection).filter_by(name="test_collection").first()
            if not collection:
                collection = Collection(
                    id="test_collection",
                    name="test_collection",
                    description="测试集合"
                )
                session.add(collection)
                session.commit()
            
            # 创建测试文档和chunk数据
            test_doc = session.query(Document).filter_by(title="测试文档").first()
            if not test_doc:
                test_doc = Document(
                    id=str(uuid.uuid4()),
                    title="测试文档",
                    content="这是一个用于测试索引构建功能的文档。它包含了一些测试内容，用于验证索引构建器是否能够正确处理文档数据。",
                    collection_id="test_collection",
                    created_at=datetime.utcnow()
                )
                session.add(test_doc)
                session.commit()
                
                # 创建测试chunk
                test_chunks = [
                    Chunk(
                        id=str(uuid.uuid4()),
                        document_id=test_doc.id,
                        content=f"这是测试chunk {i}，包含了一些测试内容用于验证索引构建功能。",
                        chunk_index=i,
                        start_pos=i * 50,
                        end_pos=(i + 1) * 50,
                        created_at=datetime.utcnow()
                    )
                    for i in range(3)
                ]
                
                for chunk in test_chunks:
                    session.add(chunk)
                session.commit()
            
            # 创建索引构建器
            builder = IndexBuilder(
                db_session=session,
                vector_store_type="faiss",
                base_index_path="/tmp/test_indices"
            )
        
            # 构建索引
            result = builder.build_collection_index("test_collection")
            
            if result.success:
                logger.info("✓ 索引构建功能测试通过")
            else:
                logger.error(f"✗ 索引构建功能测试失败: {result.error}")
                assert False, f"索引构建功能测试失败: {result.error}"
        finally:
            session.close()
        
    except Exception as e:
        logger.error(f"✗ 索引构建功能测试失败: {e}")
        assert False, f"索引构建功能测试失败: {e}"

def test_vector_retriever():
    """测试向量检索功能"""
    logger.info("开始测试向量检索功能...")
    
    try:
        from app.retrieval import VectorRetriever, RetrievalConfig
        from app.llm import get_model_manager, ModelType
        
        # 获取模型管理器
        model_manager = get_model_manager()
        
        # 获取嵌入模型提供商
        embedder = model_manager.get_provider(ModelType.EMBEDDING)
        
        # 创建检索器配置
        retrieval_config = RetrievalConfig(
            top_k=5,
            score_threshold=0.0
        )
        
        # 创建检索器（需要数据库会话）
        from app.core.database import get_session
        db_session = next(get_session())
        retriever = VectorRetriever(db_session, vector_store_type="faiss")
        
        # 测试搜索（需要先有索引数据）
        query = "测试查询文本"
        results = retriever.search(query, ["test_collection"], retrieval_config)
        
        # 验证结果格式
        assert isinstance(results, list), "检索结果应该是列表"
        
        logger.info("✓ 向量检索功能测试通过")
        
    except Exception as e:
        logger.error(f"✗ 向量检索功能测试失败: {e}")
        assert False, f"向量检索功能测试失败: {e}"

def test_mmr_selector():
    """测试MMR去冗余功能"""
    logger.info("开始测试MMR去冗余功能...")
    
    try:
        from app.retrieval import MMRSelector, MMRConfig
        from app.retrieval.vector_retriever import RetrievalResult, get_vector_retriever
        
        # 创建MMR配置
        config = MMRConfig(
            lambda_param=0.5,
            top_k=3,
            similarity_threshold=0.1
        )
        
        # 创建MMR选择器
        selector = MMRSelector()
        
        # 创建测试候选结果
        candidates = [
            RetrievalResult(
                chunk_id=f"chunk_{i}",
                vector_id=i,
                score=0.9 - i * 0.1,
                text=f"测试文本 {i}",
                metadata={"source": f"doc_{i}"},
                collection_id="test_collection"
            )
            for i in range(5)
        ]
        
        # 创建查询向量（模拟）
        import numpy as np
        query_vector = np.random.rand(768)  # 假设768维向量
        
        # 测试MMR选择
        selected = selector.select(query_vector, candidates, config)
        
        assert len(selected) <= config.top_k, "选择结果数量超过限制"
        assert all(r.score >= config.similarity_threshold for r in selected), "存在低于阈值的结果"
        
        logger.info("✓ MMR去冗余功能测试通过")
        
    except Exception as e:
        logger.error(f"✗ MMR去冗余功能测试失败: {e}")
        assert False, f"MMR去冗余功能测试失败: {e}"

def test_result_fusion():
    """测试结果融合功能"""
    logger.info("开始测试结果融合功能...")
    
    try:
        from app.retrieval import ResultFusion, FusionConfig, FusionMethod, SourceResults
        from app.retrieval.vector_retriever import RetrievalResult, get_vector_retriever
        
        # 创建融合配置
        config = FusionConfig(
            method=FusionMethod.RRF,
            rrf_k=60,
            top_k=5
        )
        
        # 创建融合器
        fusion = ResultFusion()
        
        # 创建测试源结果
        source1_results = [
            RetrievalResult(
                chunk_id=f"chunk_{i}",
                vector_id=i,
                score=0.9 - i * 0.1,
                text=f"源1文本 {i}",
                metadata={"source": "source1"},
                collection_id="test_collection"
            )
            for i in range(3)
        ]
        
        source2_results = [
            RetrievalResult(
                chunk_id=f"chunk_{i+2}",
                vector_id=i+2,
                score=0.8 - i * 0.1,
                text=f"源2文本 {i}",
                metadata={"source": "source2"},
                collection_id="test_collection"
            )
            for i in range(3)
        ]
        
        # 准备源结果
        sources = [
            SourceResults(source_id="source1", results=source1_results, weight=1.0),
            SourceResults(source_id="source2", results=source2_results, weight=0.8)
        ]
        
        # 测试融合
        fused_results = fusion.fuse(sources, config)
        
        assert len(fused_results) <= config.top_k, "融合结果数量超过限制"
        assert all(hasattr(r, 'score') for r in fused_results), "融合结果缺少分数"
        
        logger.info("✓ 结果融合功能测试通过")
        
    except Exception as e:
        logger.error(f"✗ 结果融合功能测试失败: {e}")
        assert False, f"结果融合功能测试失败: {e}"

def test_performance():
    """测试系统性能"""
    logger.info("开始测试系统性能...")
    
    try:
        from app.llm import ModernEmbedder
        
        # 使用默认嵌入器
        from app.llm import get_default_embedder
        embedder = get_default_embedder()
        
        # 性能测试数据
        test_texts = [f"性能测试文本 {i}" for i in range(100)]
        
        # 测试批量嵌入性能
        start_time = time.time()
        results = embedder.embed_texts(test_texts)
        end_time = time.time()
        
        duration = end_time - start_time
        throughput = len(test_texts) / duration
        
        logger.info(f"批量嵌入性能: {throughput:.2f} texts/sec")
        
        # 验证所有结果都成功
        assert len(results) == len(test_texts), f"嵌入结果数量不匹配: {len(results)}/{len(test_texts)}"
        assert all(isinstance(r, list) and len(r) > 0 for r in results), "嵌入结果格式错误"
        
        logger.info("✓ 系统性能测试通过")
        
    except Exception as e:
        logger.error(f"✗ 系统性能测试失败: {e}")
        assert False, f"系统性能测试失败: {e}"

def main():
    """主测试函数"""
    logger.info("开始RAG系统完整性测试...")
    
    # 首先设置嵌入模型
    logger.info("正在设置嵌入模型...")
    if not setup_embedding_model():
        logger.error("嵌入模型设置失败，无法继续测试")
        assert False, "嵌入模型设置失败，无法继续测试"
    
    test_functions = [
        ("嵌入器功能", test_embedder),
        ("FAISS存储功能", test_faiss_store),
        ("索引构建功能", test_index_builder),
        ("向量检索功能", test_vector_retriever),
        ("MMR去冗余功能", test_mmr_selector),
        ("结果融合功能", test_result_fusion),
        ("系统性能", test_performance)
    ]
    
    # 执行测试并统计结果
    test_results = {}
    for test_name, test_func in test_functions:
        try:
            test_func()
            test_results[test_name] = True
        except AssertionError as e:
            test_results[test_name] = False
            logger.error(f"测试 {test_name} 失败: {e}")
        except Exception as e:
            test_results[test_name] = False
            logger.error(f"测试 {test_name} 出现异常: {e}")
    
    # 统计测试结果
    passed = sum(1 for result in test_results.values() if result is True)
    total = len(test_results)
    
    logger.info(f"\n测试结果汇总:")
    for test_name, result in test_results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！RAG系统功能完整。")
    else:
        logger.error(f"❌ {total - passed} 个测试失败，请检查相关模块。")
        assert False, f"{total - passed} 个测试失败，请检查相关模块"

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)