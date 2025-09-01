"""向量检索器模块

实现功能：
1. 基于向量存储抽象层的检索
2. TopK检索和相似度阈值过滤
3. 多集合检索和结果融合
4. 检索结果排序和去重
5. 检索性能监控和统计
6. 支持多种向量数据库后端
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from sqlalchemy.orm import Session

from app.db.models import Collection, IndexVersion, Embedding, Chunk
from app.storage import VectorStoreFactory
from app.storage.base import VectorStore
from app.llm.model_manager import get_model_manager, ModelType
from app.utils.vector_utils import normalize_vector, cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    chunk_id: str
    vector_id: int
    score: float
    text: str
    metadata: Dict[str, Any]
    collection_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "chunk_id": self.chunk_id,
            "vector_id": self.vector_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
            "collection_id": self.collection_id
        }


@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k: int = 10
    score_threshold: float = 0.0
    max_results: int = 100
    enable_rerank: bool = False
    rerank_top_k: int = 50
    normalize_scores: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RetrievalConfig':
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


class VectorRetriever:
    """向量检索器"""
    
    def __init__(self, db_session: Session, vector_store_type: str = "faiss"):
        """
        初始化向量检索器
        
        Args:
            db_session: 数据库会话
            vector_store_type: 向量存储类型 (faiss, milvus, qdrant)
        """
        self.db = db_session
        self.vector_store_type = vector_store_type
        self.model_manager = get_model_manager()
        
        # 检索统计
        self.stats = {
            "total_queries": 0,
            "total_results": 0,
            "avg_query_time": 0.0,
            "cache_hits": 0
        }
        
    def search(self, query: str, collection_ids: List[str], 
              config: Optional[RetrievalConfig] = None) -> List[RetrievalResult]:
        """执行向量检索
        
        Args:
            query: 查询文本
            collection_ids: 要搜索的集合ID列表
            config: 检索配置
            
        Returns:
            检索结果列表
        """
        if not query.strip():
            return []
            
        if not collection_ids:
            return []
            
        config = config or RetrievalConfig()
        start_time = time.time()
        
        try:
            # 嵌入查询文本
            query_vector = self._embed_query(query)
            
            # 多集合检索
            all_results = []
            for collection_id in collection_ids:
                collection_results = self._search_collection(
                    query_vector, collection_id, config
                )
                all_results.extend(collection_results)
                
            # 合并和排序结果
            merged_results = self._merge_and_rank_results(all_results, config)
            
            # 更新统计
            query_time = time.time() - start_time
            self._update_stats(query_time, len(merged_results))
            
            logger.debug(f"检索完成: 查询='{query[:50]}...', "
                        f"集合数={len(collection_ids)}, "
                        f"结果数={len(merged_results)}, "
                        f"耗时={query_time:.3f}s")
                        
            return merged_results
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []
            
    def search_by_vector(self, query_vector: np.ndarray, collection_ids: List[str],
                        config: Optional[RetrievalConfig] = None) -> List[RetrievalResult]:
        """基于向量执行检索
        
        Args:
            query_vector: 查询向量
            collection_ids: 要搜索的集合ID列表
            config: 检索配置
            
        Returns:
            检索结果列表
        """
        if query_vector is None or len(query_vector) == 0:
            return []
            
        if not collection_ids:
            return []
            
        config = config or RetrievalConfig()
        start_time = time.time()
        
        try:
            # 多集合检索
            all_results = []
            for collection_id in collection_ids:
                collection_results = self._search_collection(
                    query_vector, collection_id, config
                )
                all_results.extend(collection_results)
                
            # 合并和排序结果
            merged_results = self._merge_and_rank_results(all_results, config)
            
            # 更新统计
            query_time = time.time() - start_time
            self._update_stats(query_time, len(merged_results))
            
            return merged_results
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
            
    def _embed_query(self, query: str) -> np.ndarray:
        """嵌入查询文本
        
        Args:
            query: 查询文本
            
        Returns:
            查询向量
        """
        try:
            # 获取嵌入模型提供商
            embedding_provider = self.model_manager.get_provider(ModelType.EMBEDDING)
            
            # 单个文本嵌入
            embeddings = embedding_provider.embed_query(query)
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"查询嵌入失败: {e}")
            raise
            
    def _search_collection(self, query_vector: np.ndarray, collection_id: str,
                          config: RetrievalConfig) -> List[RetrievalResult]:
        """在单个集合中检索
        
        Args:
            query_vector: 查询向量
            collection_id: 集合ID
            config: 检索配置
            
        Returns:
            检索结果列表
        """
        try:
            # 获取活跃的索引版本
            active_version = self._get_active_index_version(collection_id)
            if not active_version:
                logger.warning(f"集合 {collection_id} 没有活跃索引")
                return []
                
            # 创建向量存储实例
            vector_store = VectorStoreFactory.create_store(
                store_type=self.vector_store_type,
                collection_id=collection_id,
                dimension=len(query_vector)
            )
            
            # 执行向量搜索
            search_k = min(config.top_k * 2, config.max_results)  # 搜索更多结果用于后续过滤
            
            search_results = vector_store.search(
                query_vector,
                k=search_k
            )
            
            # 过滤结果
            results = []
            for search_result in search_results:
                if config.score_threshold > 0 and search_result.score < config.score_threshold:
                    continue
                    
                # 获取chunk信息
                chunk_info = self._get_chunk_by_vector_id(search_result.id, collection_id)
                if chunk_info:
                    chunk_id, text, metadata = chunk_info
                    
                    result = RetrievalResult(
                        chunk_id=chunk_id,
                        vector_id=search_result.id,
                        score=search_result.score,
                        text=text,
                        metadata=metadata,
                        collection_id=collection_id
                    )
                    results.append(result)
                    
            return results
            
        except Exception as e:
            logger.error(f"集合 {collection_id} 检索失败: {e}")
            return []
            
    def _get_active_index_version(self, collection_id: str) -> Optional[IndexVersion]:
        """获取活跃的索引版本"""
        return self.db.query(IndexVersion).filter(
            IndexVersion.collection_id == collection_id,
            IndexVersion.status == "active"
        ).first()
        
    def _get_chunk_by_vector_id(self, vector_id: int, collection_id: str) -> Optional[Tuple[str, str, Dict]]:
        """根据向量ID获取chunk信息
        
        Args:
            vector_id: 向量ID
            collection_id: 集合ID
            
        Returns:
            (chunk_id, text, metadata) 或 None
        """
        try:
            # 查询嵌入记录
            embedding = self.db.query(Embedding).filter(
                Embedding.vector_id == vector_id,
                Embedding.collection_id == collection_id
            ).first()
            
            if not embedding:
                return None
                
            # 查询chunk
            chunk = self.db.query(Chunk).filter(
                Chunk.id == embedding.chunk_id
            ).first()
            
            if not chunk:
                return None
                
            # 构建元数据
            metadata = {
                "chunk_index": chunk.chunk_index,
                "parent_id": chunk.parent_id,
                "created_at": chunk.created_at.isoformat() if chunk.created_at else None
            }
            
            # 添加chunk的元数据
            if chunk.meta_json:
                try:
                    chunk_meta = json.loads(chunk.meta_json)
                    metadata.update(chunk_meta)
                except json.JSONDecodeError:
                    pass
                    
            return chunk.id, chunk.text, metadata
            
        except Exception as e:
            logger.error(f"获取chunk信息失败 (vector_id={vector_id}): {e}")
            return None
            
    def _merge_and_rank_results(self, results: List[RetrievalResult], 
                               config: RetrievalConfig) -> List[RetrievalResult]:
        """合并和排序结果
        
        Args:
            results: 原始结果列表
            config: 检索配置
            
        Returns:
            排序后的结果列表
        """
        if not results:
            return []
            
        # 去重（基于chunk_id）
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                unique_results.append(result)
                
        # 按分数降序排序
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # 限制结果数量
        final_results = unique_results[:config.top_k]
        
        # 标准化分数
        if config.normalize_scores and final_results:
            max_score = max(r.score for r in final_results)
            min_score = min(r.score for r in final_results)
            
            if max_score > min_score:
                for result in final_results:
                    result.score = (result.score - min_score) / (max_score - min_score)
            else:
                for result in final_results:
                    result.score = 1.0
                    
        return final_results
        
    def _update_stats(self, query_time: float, result_count: int) -> None:
        """更新检索统计"""
        self.stats["total_queries"] += 1
        self.stats["total_results"] += result_count
        
        # 更新平均查询时间
        total_time = self.stats["avg_query_time"] * (self.stats["total_queries"] - 1) + query_time
        self.stats["avg_query_time"] = total_time / self.stats["total_queries"]
        
    def get_stats(self) -> Dict[str, Any]:
        """获取检索统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
        
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total_queries": 0,
            "total_results": 0,
            "avg_query_time": 0.0,
            "cache_hits": 0
        }
        
    def search_similar_chunks(self, chunk_id: str, collection_ids: List[str],
                             config: Optional[RetrievalConfig] = None) -> List[RetrievalResult]:
        """查找相似的chunk
        
        Args:
            chunk_id: 参考chunk ID
            collection_ids: 要搜索的集合ID列表
            config: 检索配置
            
        Returns:
            相似chunk列表
        """
        try:
            # 获取参考chunk的向量
            embedding = self.db.query(Embedding).filter(
                Embedding.chunk_id == chunk_id
            ).first()
            
            if not embedding:
                logger.warning(f"Chunk {chunk_id} 没有嵌入向量")
                return []
                
            # 创建向量存储实例
            vector_store = VectorStoreFactory.create_store(
                store_type=self.vector_store_type,
                collection_id=embedding.collection_id,
                dimension=0  # 维度会从存储中获取
            )
            
            # 获取向量数据
            vector_record = vector_store.get_vector(embedding.vector_id)
            if vector_record is None:
                logger.warning(f"无法获取向量 {embedding.vector_id}")
                return []
                
            # 执行相似性搜索
            results = self.search_by_vector(vector_record.vector, collection_ids, config)
            
            # 过滤掉自身
            filtered_results = [r for r in results if r.chunk_id != chunk_id]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"相似chunk搜索失败: {e}")
            return []
            
    def batch_search(self, queries: List[str], collection_ids: List[str],
                    config: Optional[RetrievalConfig] = None) -> List[List[RetrievalResult]]:
        """批量检索
        
        Args:
            queries: 查询文本列表
            collection_ids: 要搜索的集合ID列表
            config: 检索配置
            
        Returns:
            每个查询的结果列表
        """
        if not queries:
            return []
            
        config = config or RetrievalConfig()
        start_time = time.time()
        
        try:
            # 获取嵌入模型提供商
            embedding_provider = self.model_manager.get_provider(ModelType.EMBEDDING)
            
            # 批量嵌入查询
            query_vectors = embedding_provider.embed_texts(queries)
            
            # 批量检索
            all_results = []
            for query_vector in query_vectors:
                results = self.search_by_vector(query_vector, collection_ids, config)
                all_results.append(results)
                
            batch_time = time.time() - start_time
            logger.info(f"批量检索完成: {len(queries)} 个查询, 耗时 {batch_time:.3f}s")
            
            return all_results
            
        except Exception as e:
            logger.error(f"批量检索失败: {e}")
            return [[] for _ in queries]
            
    def explain_search(self, query: str, collection_ids: List[str],
                      config: Optional[RetrievalConfig] = None) -> Dict[str, Any]:
        """解释检索过程（用于调试）
        
        Args:
            query: 查询文本
            collection_ids: 集合ID列表
            config: 检索配置
            
        Returns:
            检索过程的详细信息
        """
        config = config or RetrievalConfig()
        explanation = {
            "query": query,
            "collection_ids": collection_ids,
            "config": config.__dict__,
            "steps": [],
            "results": []
        }
        
        try:
            # 步骤1: 嵌入查询
            start_time = time.time()
            query_vector = self._embed_query(query)
            embed_time = time.time() - start_time
            
            explanation["steps"].append({
                "step": "embed_query",
                "time_seconds": embed_time,
                "vector_dimension": len(query_vector)
            })
            
            # 步骤2: 多集合检索
            collection_results = {}
            for collection_id in collection_ids:
                start_time = time.time()
                results = self._search_collection(query_vector, collection_id, config)
                search_time = time.time() - start_time
                
                collection_results[collection_id] = {
                    "result_count": len(results),
                    "search_time": search_time,
                    "top_scores": [r.score for r in results[:5]]
                }
                
            explanation["steps"].append({
                "step": "collection_search",
                "collection_results": collection_results
            })
            
            # 步骤3: 合并结果
            start_time = time.time()
            all_results = []
            for collection_id in collection_ids:
                collection_results_list = self._search_collection(query_vector, collection_id, config)
                all_results.extend(collection_results_list)
                
            merged_results = self._merge_and_rank_results(all_results, config)
            merge_time = time.time() - start_time
            
            explanation["steps"].append({
                "step": "merge_and_rank",
                "time_seconds": merge_time,
                "total_candidates": len(all_results),
                "final_results": len(merged_results)
            })
            
            explanation["results"] = [r.to_dict() for r in merged_results]
            
        except Exception as e:
            explanation["error"] = str(e)
            
        return explanation


# 全局检索器实例
_vector_retriever: Optional[VectorRetriever] = None


def get_vector_retriever(db_session: Session, vector_store_type: str = "faiss") -> VectorRetriever:
    """获取向量检索器实例
    
    Args:
        db_session: 数据库会话
        vector_store_type: 向量存储类型 (faiss, milvus, qdrant)
        
    Returns:
        向量检索器实例
    """
    return VectorRetriever(db_session, vector_store_type)