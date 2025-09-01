"""Qdrant向量存储实现

基于Qdrant的向量存储实现
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import uuid

from .base import VectorStore, VectorRecord, VectorSearchResult

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, CreateCollection, PointStruct,
        Filter, FieldCondition, MatchValue, SearchRequest
    )
    from qdrant_client.http.exceptions import ResponseHandlingException
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    # 为类型注解提供占位符
    Distance = None
    Filter = None
    PointStruct = None
    FieldCondition = None
    MatchValue = None
    logger.warning("Qdrant客户端未安装，请运行: pip install qdrant-client")


class QdrantVectorStore(VectorStore):
    """Qdrant向量存储实现"""
    
    def __init__(self, config: Dict[str, Any]):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant客户端未安装，请运行: pip install qdrant-client")
            
        super().__init__(config)
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 6333)
        self.grpc_port = config.get('grpc_port', 6334)
        self.prefer_grpc = config.get('prefer_grpc', False)
        self.api_key = config.get('api_key')
        self.url = config.get('url')  # 用于云服务
        self.distance = self._get_distance_metric(config.get('distance', 'cosine'))
        
        self._client: Optional[QdrantClient] = None
        
    def initialize(self) -> None:
        """初始化Qdrant客户端"""
        try:
            if self.url:
                # 使用URL连接（通常用于云服务）
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    prefer_grpc=self.prefer_grpc
                )
            else:
                # 使用主机和端口连接
                self._client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    grpc_port=self.grpc_port,
                    prefer_grpc=self.prefer_grpc,
                    api_key=self.api_key
                )
                
            # 测试连接
            self._client.get_collections()
            
            self._is_initialized = True
            logger.info(f"Qdrant连接成功: {self.url or f'{self.host}:{self.port}'}")
            
        except Exception as e:
            logger.error(f"Qdrant初始化失败: {e}")
            raise
            
    def close(self) -> None:
        """关闭Qdrant连接"""
        if self._client:
            self._client.close()
            self._client = None
            
        self._is_initialized = False
        logger.info("Qdrant连接已关闭")
        
    def create_collection(self, collection_name: str, dimension: int,
                         metadata_schema: Optional[Dict[str, str]] = None) -> None:
        """创建Qdrant集合"""
        if not self._client:
            raise RuntimeError("Qdrant客户端未初始化")
            
        try:
            # 检查集合是否已存在
            if self.collection_exists(collection_name):
                raise ValueError(f"集合已存在: {collection_name}")
                
            # 创建集合
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=self.distance
                )
            )
            
            logger.info(f"Qdrant集合创建成功: {collection_name}, 维度: {dimension}")
            
        except Exception as e:
            logger.error(f"创建Qdrant集合失败: {e}")
            raise
            
    def delete_collection(self, collection_name: str) -> None:
        """删除Qdrant集合"""
        if not self._client:
            raise RuntimeError("Qdrant客户端未初始化")
            
        try:
            if not self.collection_exists(collection_name):
                raise ValueError(f"集合不存在: {collection_name}")
                
            self._client.delete_collection(collection_name)
            
            logger.info(f"Qdrant集合删除成功: {collection_name}")
            
        except Exception as e:
            logger.error(f"删除Qdrant集合失败: {e}")
            raise
            
    def list_collections(self) -> List[str]:
        """列出所有集合"""
        if not self._client:
            raise RuntimeError("Qdrant客户端未初始化")
            
        try:
            collections = self._client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            logger.error(f"列出Qdrant集合失败: {e}")
            return []
            
    def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        if not self._client:
            return False
            
        try:
            collections = self._client.get_collections()
            return collection_name in [collection.name for collection in collections.collections]
        except Exception as e:
            logger.error(f"检查Qdrant集合存在性失败: {e}")
            return False
            
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """获取集合信息"""
        if not self._client:
            raise RuntimeError("Qdrant客户端未初始化")
            
        if not self.collection_exists(collection_name):
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            collection_info = self._client.get_collection(collection_name)
            
            return {
                'name': collection_name,
                'vector_count': collection_info.points_count,
                'dimension': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance.value,
                'status': collection_info.status.value,
                'optimizer_status': collection_info.optimizer_status
            }
            
        except Exception as e:
            logger.error(f"获取Qdrant集合信息失败: {e}")
            raise
            
    def insert_vectors(self, collection_name: str, vectors: List[VectorRecord]) -> List[str]:
        """插入向量到Qdrant"""
        if not self._client:
            raise RuntimeError("Qdrant客户端未初始化")
            
        if not vectors:
            return []
            
        if not self.collection_exists(collection_name):
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            # 准备点数据
            points = []
            inserted_ids = []
            
            for record in vectors:
                # 使用提供的ID或生成UUID
                point_id = record.id if record.id else str(uuid.uuid4())
                
                # 验证向量维度
                vector = record.vector
                if self.normalize_vectors:
                    vector = self.normalize_vector(vector)
                    
                point = PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=record.metadata or {}
                )
                
                points.append(point)
                inserted_ids.append(point_id)
                
            # 批量插入
            self._client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"向Qdrant插入向量成功: {len(inserted_ids)} 个向量到集合 {collection_name}")
            
            return inserted_ids
            
        except Exception as e:
            logger.error(f"向Qdrant插入向量失败: {e}")
            raise
            
    def update_vectors(self, collection_name: str, vectors: List[VectorRecord]) -> None:
        """更新向量（Qdrant支持upsert操作）"""
        # Qdrant的upsert操作可以直接更新
        self.insert_vectors(collection_name, vectors)
        
    def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> None:
        """从Qdrant删除向量"""
        if not self._client:
            raise RuntimeError("Qdrant客户端未初始化")
            
        if not vector_ids:
            return
            
        if not self.collection_exists(collection_name):
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            # 批量删除
            self._client.delete(
                collection_name=collection_name,
                points_selector=vector_ids
            )
            
            logger.info(f"从Qdrant删除向量成功: {len(vector_ids)} 个向量从集合 {collection_name}")
            
        except Exception as e:
            logger.error(f"从Qdrant删除向量失败: {e}")
            raise
            
    def get_vector(self, collection_name: str, vector_id: str) -> Optional[VectorRecord]:
        """从Qdrant获取单个向量"""
        if not self._client:
            raise RuntimeError("Qdrant客户端未初始化")
            
        if not self.collection_exists(collection_name):
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            # 获取点
            points = self._client.retrieve(
                collection_name=collection_name,
                ids=[vector_id],
                with_vectors=True,
                with_payload=True
            )
            
            if not points:
                return None
                
            point = points[0]
            
            return VectorRecord(
                id=str(point.id),
                vector=np.array(point.vector),
                metadata=point.payload or {}
            )
            
        except Exception as e:
            logger.error(f"从Qdrant获取向量失败: {e}")
            return None
            
    def search_vectors(self, collection_name: str, query_vector: np.ndarray,
                      top_k: int = 10, filter_expr: Optional[str] = None,
                      **kwargs) -> List[VectorSearchResult]:
        """在Qdrant中搜索相似向量"""
        if not self._client:
            raise RuntimeError("Qdrant客户端未初始化")
            
        if not self.collection_exists(collection_name):
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            # 归一化查询向量
            if self.normalize_vectors:
                query_vector = self.normalize_vector(query_vector)
                
            # 构建过滤器
            query_filter = self._build_filter(filter_expr) if filter_expr else None
            
            # 执行搜索
            results = self._client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True
            )
            
            # 处理结果
            search_results = []
            for result in results:
                search_results.append(VectorSearchResult(
                    id=str(result.id),
                    score=float(result.score),
                    metadata=result.payload or {}
                ))
                
            return search_results
            
        except Exception as e:
            logger.error(f"Qdrant向量搜索失败: {e}")
            raise
            
    def batch_search_vectors(self, collection_name: str, query_vectors: List[np.ndarray],
                           top_k: int = 10, filter_expr: Optional[str] = None,
                           **kwargs) -> List[List[VectorSearchResult]]:
        """在Qdrant中批量搜索相似向量"""
        if not self._client:
            raise RuntimeError("Qdrant客户端未初始化")
            
        if not self.collection_exists(collection_name):
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            # 归一化查询向量
            normalized_vectors = []
            for vector in query_vectors:
                if self.normalize_vectors:
                    normalized_vectors.append(self.normalize_vector(vector))
                else:
                    normalized_vectors.append(vector)
                    
            # 构建过滤器
            query_filter = self._build_filter(filter_expr) if filter_expr else None
            
            # 构建搜索请求
            search_requests = []
            for vector in normalized_vectors:
                search_requests.append(SearchRequest(
                    vector=vector.tolist(),
                    limit=top_k,
                    filter=query_filter,
                    with_payload=True
                ))
                
            # 执行批量搜索
            batch_results = self._client.search_batch(
                collection_name=collection_name,
                requests=search_requests
            )
            
            # 处理结果
            all_results = []
            for results in batch_results:
                search_results = []
                for result in results:
                    search_results.append(VectorSearchResult(
                        id=str(result.id),
                        score=float(result.score),
                        metadata=result.payload or {}
                    ))
                all_results.append(search_results)
                
            return all_results
            
        except Exception as e:
            logger.error(f"Qdrant批量向量搜索失败: {e}")
            raise
            
    def count_vectors(self, collection_name: str, filter_expr: Optional[str] = None) -> int:
        """统计Qdrant集合中的向量数量"""
        if not self._client:
            raise RuntimeError("Qdrant客户端未初始化")
            
        if not self.collection_exists(collection_name):
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            if filter_expr:
                # 使用过滤器统计
                query_filter = self._build_filter(filter_expr)
                result = self._client.count(
                    collection_name=collection_name,
                    count_filter=query_filter
                )
                return result.count
            else:
                # 获取总数
                collection_info = self._client.get_collection(collection_name)
                return collection_info.points_count
                
        except Exception as e:
            logger.error(f"统计Qdrant向量数量失败: {e}")
            raise
            
    def _get_distance_metric(self, distance: str) -> Distance:
        """获取距离度量"""
        distance_map = {
            'cosine': Distance.COSINE,
            'euclidean': Distance.EUCLID,
            'dot': Distance.DOT,
            'manhattan': Distance.MANHATTAN
        }
        
        return distance_map.get(distance.lower(), Distance.COSINE)
        
    def _build_filter(self, filter_expr: str) -> Optional[Filter]:
        """构建Qdrant过滤器
        
        这是一个简化的实现，实际项目中可能需要更复杂的过滤逻辑
        """
        try:
            if '=' in filter_expr:
                key, value = filter_expr.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                
                # 尝试转换为数字
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass
                    
                return Filter(
                    must=[
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    ]
                )
                
        except Exception as e:
            logger.warning(f"构建过滤器失败: {e}")
            
        return None


# 注册Qdrant存储类型
if QDRANT_AVAILABLE:
    from .base import VectorStoreFactory
    VectorStoreFactory.register('qdrant', QdrantVectorStore)