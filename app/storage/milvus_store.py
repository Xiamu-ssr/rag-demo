"""Milvus向量存储实现

基于Milvus的向量存储实现
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from .base import VectorStore, VectorRecord, VectorSearchResult

logger = logging.getLogger(__name__)

try:
    from pymilvus import (
        connections, Collection, CollectionSchema, FieldSchema, DataType,
        utility, MilvusException
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    Collection = None  # 为类型注解提供占位符
    logger.warning("Milvus客户端未安装，请运行: pip install pymilvus")


class MilvusVectorStore(VectorStore):
    """Milvus向量存储实现"""
    
    def __init__(self, config: Dict[str, Any]):
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus客户端未安装，请运行: pip install pymilvus")
            
        super().__init__(config)
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 19530)
        self.user = config.get('user', '')
        self.password = config.get('password', '')
        self.connection_alias = config.get('connection_alias', 'default')
        self.index_type = config.get('index_type', 'IVF_FLAT')
        self.metric_type = config.get('metric_type', 'IP')  # Inner Product
        self.nlist = config.get('nlist', 1024)  # IVF参数
        
        self._collections: Dict[str, Collection] = {}
        
    def initialize(self) -> None:
        """初始化Milvus连接"""
        try:
            # 建立连接
            connections.connect(
                alias=self.connection_alias,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            
            # 加载已有集合
            self._load_existing_collections()
            
            self._is_initialized = True
            logger.info(f"Milvus连接成功: {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Milvus初始化失败: {e}")
            raise
            
    def close(self) -> None:
        """关闭Milvus连接"""
        try:
            # 释放集合
            for collection in self._collections.values():
                collection.release()
                
            # 断开连接
            connections.disconnect(alias=self.connection_alias)
            
            self._collections.clear()
            self._is_initialized = False
            logger.info("Milvus连接已关闭")
            
        except Exception as e:
            logger.error(f"关闭Milvus连接失败: {e}")
            
    def create_collection(self, collection_name: str, dimension: int,
                         metadata_schema: Optional[Dict[str, str]] = None) -> None:
        """创建Milvus集合"""
        if collection_name in self._collections:
            raise ValueError(f"集合已存在: {collection_name}")
            
        try:
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            
            # 添加元数据字段
            if metadata_schema:
                for field_name, field_type in metadata_schema.items():
                    if field_type.lower() == 'string':
                        fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=1000))
                    elif field_type.lower() == 'int':
                        fields.append(FieldSchema(name=field_name, dtype=DataType.INT64))
                    elif field_type.lower() == 'float':
                        fields.append(FieldSchema(name=field_name, dtype=DataType.DOUBLE))
                    elif field_type.lower() == 'bool':
                        fields.append(FieldSchema(name=field_name, dtype=DataType.BOOL))
                        
            # 创建集合schema
            schema = CollectionSchema(
                fields=fields,
                description=f"Collection for {collection_name}"
            )
            
            # 创建集合
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.connection_alias
            )
            
            # 创建索引
            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {"nlist": self.nlist}
            }
            
            collection.create_index(
                field_name="vector",
                index_params=index_params
            )
            
            # 加载集合到内存
            collection.load()
            
            self._collections[collection_name] = collection
            
            logger.info(f"Milvus集合创建成功: {collection_name}, 维度: {dimension}")
            
        except Exception as e:
            logger.error(f"创建Milvus集合失败: {e}")
            raise
            
    def delete_collection(self, collection_name: str) -> None:
        """删除Milvus集合"""
        if collection_name not in self._collections:
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            collection = self._collections[collection_name]
            collection.release()
            collection.drop()
            
            del self._collections[collection_name]
            
            logger.info(f"Milvus集合删除成功: {collection_name}")
            
        except Exception as e:
            logger.error(f"删除Milvus集合失败: {e}")
            raise
            
    def list_collections(self) -> List[str]:
        """列出所有集合"""
        try:
            return utility.list_collections(using=self.connection_alias)
        except Exception as e:
            logger.error(f"列出集合失败: {e}")
            return list(self._collections.keys())
            
    def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        try:
            return utility.has_collection(collection_name, using=self.connection_alias)
        except Exception as e:
            logger.error(f"检查集合存在性失败: {e}")
            return collection_name in self._collections
            
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """获取集合信息"""
        if collection_name not in self._collections:
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            collection = self._collections[collection_name]
            stats = collection.get_stats()
            
            return {
                'name': collection_name,
                'vector_count': int(stats.get('row_count', 0)),
                'dimension': self._get_vector_dimension(collection),
                'index_type': self.index_type,
                'metric_type': self.metric_type,
                'schema': collection.schema.to_dict()
            }
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            raise
            
    def insert_vectors(self, collection_name: str, vectors: List[VectorRecord]) -> List[str]:
        """插入向量到Milvus"""
        if collection_name not in self._collections:
            raise ValueError(f"集合不存在: {collection_name}")
            
        if not vectors:
            return []
            
        try:
            collection = self._collections[collection_name]
            
            # 准备数据
            ids = [record.id for record in vectors]
            vector_data = [record.vector.tolist() for record in vectors]
            
            # 准备插入数据
            insert_data = {
                "id": ids,
                "vector": vector_data
            }
            
            # 添加元数据字段
            schema_fields = {field.name for field in collection.schema.fields}
            for record in vectors:
                if record.metadata:
                    for key, value in record.metadata.items():
                        if key in schema_fields:
                            if key not in insert_data:
                                insert_data[key] = []
                            insert_data[key].append(value)
                        
            # 确保所有字段都有相同长度的数据
            for field in collection.schema.fields:
                if field.name not in insert_data and field.name not in ['id', 'vector']:
                    insert_data[field.name] = [None] * len(ids)
                    
            # 执行插入
            result = collection.insert(insert_data)
            
            # 刷新数据到磁盘
            collection.flush()
            
            logger.info(f"向Milvus插入向量成功: {len(ids)} 个向量到集合 {collection_name}")
            
            return ids
            
        except Exception as e:
            logger.error(f"向Milvus插入向量失败: {e}")
            raise
            
    def update_vectors(self, collection_name: str, vectors: List[VectorRecord]) -> None:
        """更新向量（Milvus通过删除后插入实现）"""
        if collection_name not in self._collections:
            raise ValueError(f"集合不存在: {collection_name}")
            
        # Milvus不支持原地更新，需要删除后重新插入
        vector_ids = [record.id for record in vectors]
        self.delete_vectors(collection_name, vector_ids)
        self.insert_vectors(collection_name, vectors)
        
    def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> None:
        """从Milvus删除向量"""
        if collection_name not in self._collections:
            raise ValueError(f"集合不存在: {collection_name}")
            
        if not vector_ids:
            return
            
        try:
            collection = self._collections[collection_name]
            
            # 构建删除表达式
            id_list = "', '".join(vector_ids)
            expr = f"id in ['{id_list}']"
            
            # 执行删除
            collection.delete(expr)
            
            # 刷新数据
            collection.flush()
            
            logger.info(f"从Milvus删除向量成功: {len(vector_ids)} 个向量从集合 {collection_name}")
            
        except Exception as e:
            logger.error(f"从Milvus删除向量失败: {e}")
            raise
            
    def get_vector(self, collection_name: str, vector_id: str) -> Optional[VectorRecord]:
        """从Milvus获取单个向量"""
        if collection_name not in self._collections:
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            collection = self._collections[collection_name]
            
            # 查询向量
            expr = f"id == '{vector_id}'"
            results = collection.query(
                expr=expr,
                output_fields=["*"]
            )
            
            if not results:
                return None
                
            result = results[0]
            
            # 提取元数据
            metadata = {}
            for key, value in result.items():
                if key not in ['id', 'vector']:
                    metadata[key] = value
                    
            return VectorRecord(
                id=result['id'],
                vector=np.array(result['vector']),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"从Milvus获取向量失败: {e}")
            return None
            
    def search_vectors(self, collection_name: str, query_vector: np.ndarray,
                      top_k: int = 10, filter_expr: Optional[str] = None,
                      **kwargs) -> List[VectorSearchResult]:
        """在Milvus中搜索相似向量"""
        if collection_name not in self._collections:
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            collection = self._collections[collection_name]
            
            # 准备搜索参数
            search_params = {
                "metric_type": self.metric_type,
                "params": {"nprobe": min(16, self.nlist)}
            }
            
            # 执行搜索
            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["*"]
            )
            
            # 处理结果
            search_results = []
            for hit in results[0]:
                # 提取元数据
                metadata = {}
                if hasattr(hit, 'entity'):
                    for key, value in hit.entity.fields.items():
                        if key not in ['id', 'vector']:
                            metadata[key] = value
                            
                search_results.append(VectorSearchResult(
                    id=hit.id,
                    score=float(hit.score),
                    metadata=metadata
                ))
                
            return search_results
            
        except Exception as e:
            logger.error(f"Milvus向量搜索失败: {e}")
            raise
            
    def batch_search_vectors(self, collection_name: str, query_vectors: List[np.ndarray],
                           top_k: int = 10, filter_expr: Optional[str] = None,
                           **kwargs) -> List[List[VectorSearchResult]]:
        """在Milvus中批量搜索相似向量"""
        if collection_name not in self._collections:
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            collection = self._collections[collection_name]
            
            # 准备搜索参数
            search_params = {
                "metric_type": self.metric_type,
                "params": {"nprobe": min(16, self.nlist)}
            }
            
            # 执行批量搜索
            query_data = [vector.tolist() for vector in query_vectors]
            results = collection.search(
                data=query_data,
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["*"]
            )
            
            # 处理结果
            all_results = []
            for query_result in results:
                search_results = []
                for hit in query_result:
                    # 提取元数据
                    metadata = {}
                    if hasattr(hit, 'entity'):
                        for key, value in hit.entity.fields.items():
                            if key not in ['id', 'vector']:
                                metadata[key] = value
                                
                    search_results.append(VectorSearchResult(
                        id=hit.id,
                        score=float(hit.score),
                        metadata=metadata
                    ))
                    
                all_results.append(search_results)
                
            return all_results
            
        except Exception as e:
            logger.error(f"Milvus批量向量搜索失败: {e}")
            raise
            
    def count_vectors(self, collection_name: str, filter_expr: Optional[str] = None) -> int:
        """统计Milvus集合中的向量数量"""
        if collection_name not in self._collections:
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            collection = self._collections[collection_name]
            
            if filter_expr:
                # 使用查询统计
                results = collection.query(
                    expr=filter_expr,
                    output_fields=["id"]
                )
                return len(results)
            else:
                # 获取总数
                stats = collection.get_stats()
                return int(stats.get('row_count', 0))
                
        except Exception as e:
            logger.error(f"统计Milvus向量数量失败: {e}")
            raise
            
    def _get_vector_dimension(self, collection: Collection) -> int:
        """获取向量维度"""
        for field in collection.schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                return field.params.get('dim', 0)
        return 0
        
    def _load_existing_collections(self) -> None:
        """加载已存在的集合"""
        try:
            collection_names = utility.list_collections(using=self.connection_alias)
            
            for name in collection_names:
                try:
                    collection = Collection(name, using=self.connection_alias)
                    collection.load()
                    self._collections[name] = collection
                    logger.info(f"加载Milvus集合: {name}")
                except Exception as e:
                    logger.warning(f"加载Milvus集合失败 {name}: {e}")
                    
        except Exception as e:
            logger.error(f"加载已存在Milvus集合失败: {e}")


# 注册Milvus存储类型
if MILVUS_AVAILABLE:
    from .base import VectorStoreFactory
    VectorStoreFactory.register('milvus', MilvusVectorStore)