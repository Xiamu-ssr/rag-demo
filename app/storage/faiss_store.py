"""FAISS向量存储实现

基于FAISS库的向量存储实现
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

from .base import VectorStore, VectorRecord, VectorSearchResult

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """FAISS向量存储实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.storage_path = config.get('storage_path', './faiss_storage')
        self.index_type = config.get('index_type', 'IndexFlatIP')  # 默认使用内积索引
        self.normalize_vectors = config.get('normalize_vectors', True)
        
        # 内存中的索引和元数据
        self._indexes: Dict[str, faiss.Index] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}  # collection_name -> {id -> metadata}
        self._id_mapping: Dict[str, Dict[str, int]] = {}  # collection_name -> {external_id -> internal_id}
        self._reverse_id_mapping: Dict[str, Dict[int, str]] = {}  # collection_name -> {internal_id -> external_id}
        self._collection_info: Dict[str, Dict[str, Any]] = {}  # 集合信息
        
    def initialize(self) -> None:
        """初始化FAISS存储"""
        try:
            # 创建存储目录
            os.makedirs(self.storage_path, exist_ok=True)
            
            # 加载已有的集合
            self._load_existing_collections()
            
            self._is_initialized = True
            logger.info(f"FAISS存储初始化成功，存储路径: {self.storage_path}")
            
        except Exception as e:
            logger.error(f"FAISS存储初始化失败: {e}")
            raise
            
    def close(self) -> None:
        """关闭存储"""
        # 保存所有集合
        for collection_name in list(self._indexes.keys()):
            self._save_collection(collection_name)
            
        # 清理内存
        self._indexes.clear()
        self._metadata.clear()
        self._id_mapping.clear()
        self._reverse_id_mapping.clear()
        self._collection_info.clear()
        
        self._is_initialized = False
        logger.info("FAISS存储已关闭")
        
    def create_collection(self, collection_name: str, dimension: int, 
                         metadata_schema: Optional[Dict[str, str]] = None) -> None:
        """创建集合"""
        if collection_name in self._indexes:
            raise ValueError(f"集合已存在: {collection_name}")
            
        try:
            # 创建FAISS索引
            if self.index_type == 'IndexFlatIP':
                index = faiss.IndexFlatIP(dimension)
            elif self.index_type == 'IndexFlatL2':
                index = faiss.IndexFlatL2(dimension)
            elif self.index_type == 'IndexIVFFlat':
                # IVF索引需要训练
                nlist = min(100, max(1, dimension // 10))  # 聚类中心数量
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            elif self.index_type == 'IndexHNSW':
                # HNSW索引
                index = faiss.IndexHNSWFlat(dimension, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 50
            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")
                
            # 初始化集合数据结构
            self._indexes[collection_name] = index
            self._metadata[collection_name] = {}
            self._id_mapping[collection_name] = {}
            self._reverse_id_mapping[collection_name] = {}
            self._collection_info[collection_name] = {
                'dimension': dimension,
                'index_type': self.index_type,
                'metadata_schema': metadata_schema or {},
                'created_at': np.datetime64('now').astype(str),
                'vector_count': 0
            }
            
            # 保存集合
            self._save_collection(collection_name)
            
            logger.info(f"集合创建成功: {collection_name}, 维度: {dimension}, 索引类型: {self.index_type}")
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise
            
    def delete_collection(self, collection_name: str) -> None:
        """删除集合"""
        if collection_name not in self._indexes:
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            # 删除内存数据
            del self._indexes[collection_name]
            del self._metadata[collection_name]
            del self._id_mapping[collection_name]
            del self._reverse_id_mapping[collection_name]
            del self._collection_info[collection_name]
            
            # 删除文件
            collection_dir = os.path.join(self.storage_path, collection_name)
            if os.path.exists(collection_dir):
                import shutil
                shutil.rmtree(collection_dir)
                
            logger.info(f"集合删除成功: {collection_name}")
            
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            raise
            
    def list_collections(self) -> List[str]:
        """列出所有集合"""
        return list(self._indexes.keys())
        
    def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        return collection_name in self._indexes
        
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """获取集合信息"""
        if collection_name not in self._indexes:
            raise ValueError(f"集合不存在: {collection_name}")
            
        info = self._collection_info[collection_name].copy()
        info['vector_count'] = self._indexes[collection_name].ntotal
        return info
        
    def insert_vectors(self, collection_name: str, vectors: List[VectorRecord]) -> List[str]:
        """插入向量"""
        if collection_name not in self._indexes:
            raise ValueError(f"集合不存在: {collection_name}")
            
        if not vectors:
            return []
            
        try:
            index = self._indexes[collection_name]
            metadata_dict = self._metadata[collection_name]
            id_mapping = self._id_mapping[collection_name]
            reverse_id_mapping = self._reverse_id_mapping[collection_name]
            
            # 准备向量数据
            vector_data = []
            inserted_ids = []
            
            for record in vectors:
                # 检查ID是否已存在
                if record.id in id_mapping:
                    logger.warning(f"向量ID已存在，跳过: {record.id}")
                    continue
                    
                # 验证向量维度
                expected_dim = self._collection_info[collection_name]['dimension']
                if record.vector.shape[-1] != expected_dim:
                    raise ValueError(f"向量维度不匹配: 期望 {expected_dim}, 实际 {record.vector.shape[-1]}")
                    
                # 归一化向量（如果需要）
                vector = record.vector.copy()
                if self.normalize_vectors:
                    vector = self.normalize_vector(vector)
                    
                vector_data.append(vector)
                inserted_ids.append(record.id)
                
                # 存储元数据和ID映射
                internal_id = index.ntotal + len(vector_data) - 1
                id_mapping[record.id] = internal_id
                reverse_id_mapping[internal_id] = record.id
                metadata_dict[record.id] = record.metadata or {}
                
            if vector_data:
                # 转换为numpy数组并添加到索引
                vectors_array = np.array(vector_data, dtype=np.float32)
                
                # 对于需要训练的索引类型
                if hasattr(index, 'is_trained') and not index.is_trained:
                    if vectors_array.shape[0] >= 100:  # 需要足够的训练数据
                        index.train(vectors_array)
                    else:
                        logger.warning(f"训练数据不足，使用Flat索引替代")
                        # 替换为Flat索引
                        dimension = vectors_array.shape[1]
                        new_index = faiss.IndexFlatIP(dimension)
                        self._indexes[collection_name] = new_index
                        index = new_index
                        
                index.add(vectors_array)
                
                # 保存集合
                self._save_collection(collection_name)
                
                logger.info(f"插入向量成功: {len(inserted_ids)} 个向量到集合 {collection_name}")
                
            return inserted_ids
            
        except Exception as e:
            logger.error(f"插入向量失败: {e}")
            raise
            
    def update_vectors(self, collection_name: str, vectors: List[VectorRecord]) -> None:
        """更新向量（FAISS不支持原地更新，需要重建索引）"""
        if collection_name not in self._indexes:
            raise ValueError(f"集合不存在: {collection_name}")
            
        # FAISS不支持原地更新，需要删除后重新插入
        vector_ids = [record.id for record in vectors]
        self.delete_vectors(collection_name, vector_ids)
        self.insert_vectors(collection_name, vectors)
        
    def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> None:
        """删除向量（FAISS不支持原地删除，需要重建索引）"""
        if collection_name not in self._indexes:
            raise ValueError(f"集合不存在: {collection_name}")
            
        if not vector_ids:
            return
            
        try:
            # 获取所有现有向量（除了要删除的）
            existing_vectors = []
            metadata_dict = self._metadata[collection_name]
            id_mapping = self._id_mapping[collection_name]
            reverse_id_mapping = self._reverse_id_mapping[collection_name]
            
            for external_id, metadata in metadata_dict.items():
                if external_id not in vector_ids:
                    internal_id = id_mapping[external_id]
                    vector = self._indexes[collection_name].reconstruct(internal_id)
                    existing_vectors.append(VectorRecord(
                        id=external_id,
                        vector=vector,
                        metadata=metadata
                    ))
                    
            # 重建集合
            collection_info = self._collection_info[collection_name]
            dimension = collection_info['dimension']
            metadata_schema = collection_info.get('metadata_schema')
            
            # 删除旧集合
            del self._indexes[collection_name]
            del self._metadata[collection_name]
            del self._id_mapping[collection_name]
            del self._reverse_id_mapping[collection_name]
            
            # 重新创建集合
            self.create_collection(collection_name, dimension, metadata_schema)
            
            # 重新插入向量
            if existing_vectors:
                self.insert_vectors(collection_name, existing_vectors)
                
            logger.info(f"删除向量成功: {len(vector_ids)} 个向量从集合 {collection_name}")
            
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            raise
            
    def get_vector(self, collection_name: str, vector_id: str) -> Optional[VectorRecord]:
        """获取单个向量"""
        if collection_name not in self._indexes:
            raise ValueError(f"集合不存在: {collection_name}")
            
        id_mapping = self._id_mapping[collection_name]
        if vector_id not in id_mapping:
            return None
            
        try:
            internal_id = id_mapping[vector_id]
            vector = self._indexes[collection_name].reconstruct(internal_id)
            metadata = self._metadata[collection_name].get(vector_id, {})
            
            return VectorRecord(
                id=vector_id,
                vector=vector,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"获取向量失败: {e}")
            return None
            
    def search_vectors(self, collection_name: str, query_vector: np.ndarray, 
                      top_k: int = 10, filter_expr: Optional[str] = None,
                      **kwargs) -> List[VectorSearchResult]:
        """向量相似性搜索"""
        if collection_name not in self._indexes:
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            index = self._indexes[collection_name]
            reverse_id_mapping = self._reverse_id_mapping[collection_name]
            metadata_dict = self._metadata[collection_name]
            
            # 归一化查询向量
            if self.normalize_vectors:
                query_vector = self.normalize_vector(query_vector)
                
            # 执行搜索
            query_vector = query_vector.reshape(1, -1).astype(np.float32)
            scores, indices = index.search(query_vector, min(top_k, index.ntotal))
            
            # 构建结果
            results = []
            for i, (score, internal_id) in enumerate(zip(scores[0], indices[0])):
                if internal_id == -1:  # FAISS返回-1表示无效结果
                    break
                    
                external_id = reverse_id_mapping.get(internal_id)
                if external_id is None:
                    continue
                    
                # 应用过滤器（简单实现）
                metadata = metadata_dict.get(external_id, {})
                if filter_expr and not self._apply_filter(metadata, filter_expr):
                    continue
                    
                results.append(VectorSearchResult(
                    id=external_id,
                    score=float(score),
                    metadata=metadata
                ))
                
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            raise
            
    def batch_search_vectors(self, collection_name: str, query_vectors: List[np.ndarray],
                           top_k: int = 10, filter_expr: Optional[str] = None,
                           **kwargs) -> List[List[VectorSearchResult]]:
        """批量向量相似性搜索"""
        if collection_name not in self._indexes:
            raise ValueError(f"集合不存在: {collection_name}")
            
        try:
            index = self._indexes[collection_name]
            reverse_id_mapping = self._reverse_id_mapping[collection_name]
            metadata_dict = self._metadata[collection_name]
            
            # 准备查询向量
            query_array = np.array(query_vectors, dtype=np.float32)
            if self.normalize_vectors:
                query_array = np.array([self.normalize_vector(v) for v in query_vectors], dtype=np.float32)
                
            # 执行批量搜索
            scores, indices = index.search(query_array, min(top_k, index.ntotal))
            
            # 构建结果
            all_results = []
            for query_idx in range(len(query_vectors)):
                results = []
                for i, (score, internal_id) in enumerate(zip(scores[query_idx], indices[query_idx])):
                    if internal_id == -1:
                        break
                        
                    external_id = reverse_id_mapping.get(internal_id)
                    if external_id is None:
                        continue
                        
                    metadata = metadata_dict.get(external_id, {})
                    if filter_expr and not self._apply_filter(metadata, filter_expr):
                        continue
                        
                    results.append(VectorSearchResult(
                        id=external_id,
                        score=float(score),
                        metadata=metadata
                    ))
                    
                all_results.append(results[:top_k])
                
            return all_results
            
        except Exception as e:
            logger.error(f"批量向量搜索失败: {e}")
            raise
            
    def count_vectors(self, collection_name: str, filter_expr: Optional[str] = None) -> int:
        """统计向量数量"""
        if collection_name not in self._indexes:
            raise ValueError(f"集合不存在: {collection_name}")
            
        if filter_expr is None:
            return self._indexes[collection_name].ntotal
            
        # 应用过滤器统计
        count = 0
        metadata_dict = self._metadata[collection_name]
        for metadata in metadata_dict.values():
            if self._apply_filter(metadata, filter_expr):
                count += 1
                
        return count
        
    def _apply_filter(self, metadata: Dict[str, Any], filter_expr: str) -> bool:
        """应用简单的过滤表达式
        
        这是一个简化的实现，实际项目中可能需要更复杂的过滤逻辑
        """
        try:
            # 简单的键值匹配
            if '=' in filter_expr:
                key, value = filter_expr.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                return str(metadata.get(key, '')) == value
            return True
        except Exception:
            return True
            
    def _save_collection(self, collection_name: str) -> None:
        """保存集合到磁盘"""
        try:
            collection_dir = os.path.join(self.storage_path, collection_name)
            os.makedirs(collection_dir, exist_ok=True)
            
            # 保存FAISS索引
            index_path = os.path.join(collection_dir, 'index.faiss')
            faiss.write_index(self._indexes[collection_name], index_path)
            
            # 保存元数据
            metadata_path = os.path.join(collection_dir, 'metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self._metadata[collection_name],
                    'id_mapping': self._id_mapping[collection_name],
                    'reverse_id_mapping': self._reverse_id_mapping[collection_name],
                    'collection_info': self._collection_info[collection_name]
                }, f)
                
        except Exception as e:
            logger.error(f"保存集合失败: {e}")
            raise
            
    def _load_collection(self, collection_name: str) -> None:
        """从磁盘加载集合"""
        try:
            collection_dir = os.path.join(self.storage_path, collection_name)
            
            # 加载FAISS索引
            index_path = os.path.join(collection_dir, 'index.faiss')
            if os.path.exists(index_path):
                self._indexes[collection_name] = faiss.read_index(index_path)
                
            # 加载元数据
            metadata_path = os.path.join(collection_dir, 'metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self._metadata[collection_name] = data.get('metadata', {})
                    self._id_mapping[collection_name] = data.get('id_mapping', {})
                    self._reverse_id_mapping[collection_name] = data.get('reverse_id_mapping', {})
                    self._collection_info[collection_name] = data.get('collection_info', {})
                    
                logger.info(f"集合加载成功: {collection_name}")
            else:
                logger.warning(f"集合元数据文件不存在: {collection_name}")
                
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            raise
            
    def _load_existing_collections(self) -> None:
        """加载所有已存在的集合"""
        if not os.path.exists(self.storage_path):
            return
            
        try:
            for item in os.listdir(self.storage_path):
                collection_dir = os.path.join(self.storage_path, item)
                if os.path.isdir(collection_dir):
                    index_path = os.path.join(collection_dir, 'index.faiss')
                    if os.path.exists(index_path):
                        self._load_collection(item)
                        
        except Exception as e:
            logger.error(f"加载已存在集合失败: {e}")
            

# 注册FAISS存储类型
from .base import VectorStoreFactory
VectorStoreFactory.register('faiss', FAISSVectorStore)