"""索引构建器模块

实现功能：
1. 简化的索引构建流程
2. 构建状态跟踪和错误处理
3. 支持增量构建和全量重建
4. 与数据库模型和新向量存储架构的集成
"""

import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sqlalchemy.orm import Session

from app.db.models import Collection, IndexVersion, Embedding, Chunk, Document, Parent
from app.storage import VectorStoreFactory, VectorStore
from app.storage.base import VectorRecord
from app.llm.model_manager import get_model_manager, ModelType

logger = logging.getLogger(__name__)


class IndexStatus(str, Enum):
    """索引状态枚举"""
    BUILDING = "building"
    ACTIVE = "active"
    FAILED = "failed"
    INACTIVE = "inactive"


class IndexBuildResult:
    """索引构建结果"""
    
    def __init__(self, success: bool, version_id: Optional[int] = None, 
                 error: Optional[str] = None, stats: Optional[Dict] = None):
        self.success = success
        self.version_id = version_id
        self.error = error
        self.stats = stats or {}
        
    def __repr__(self):
        return f"IndexBuildResult(success={self.success}, version_id={self.version_id})"


class IndexBuilder:
    """索引构建器"""
    
    def __init__(self, db_session: Session, vector_store_type: str = "faiss", 
                 base_index_path: str = "/data/indices"):
        """
        初始化索引构建器
        
        Args:
            db_session: 数据库会话
            vector_store_type: 向量存储类型
            base_index_path: 索引文件基础路径
        """
        self.db = db_session
        self.base_index_path = Path(base_index_path)
        self.vector_store_type = vector_store_type
        self.model_manager = get_model_manager()
        
        # 确保基础目录存在
        self.base_index_path.mkdir(parents=True, exist_ok=True)
        
    def build_collection_index(self, collection_id: str, force_rebuild: bool = False) -> IndexBuildResult:
        """构建集合索引
        
        Args:
            collection_id: 集合ID
            force_rebuild: 是否强制重建
            
        Returns:
            构建结果
        """
        try:
            logger.info(f"开始构建集合索引: {collection_id}")
            
            # 获取集合信息
            collection = self.db.query(Collection).filter(Collection.id == collection_id).first()
            if not collection:
                raise ValueError(f"集合不存在: {collection_id}")
                
            # 检查是否需要构建
            if not force_rebuild and self._has_active_index(collection_id):
                logger.info(f"集合 {collection_id} 已有活跃索引，跳过构建")
                active_version = self._get_active_version(collection_id)
                return IndexBuildResult(True, active_version.id if active_version else None)
                
            # 创建新的索引版本记录
            version = self._create_index_version(collection)
            
            try:
                # 获取所有需要索引的chunk
                chunks_data = self._get_chunks_for_indexing(collection_id)
                
                if not chunks_data:
                    logger.warning(f"集合 {collection_id} 没有可索引的chunk")
                    self._update_version_status(version.id, IndexStatus.FAILED, "没有可索引的数据")
                    return IndexBuildResult(False, version.id, "没有可索引的数据")
                    
                # 执行嵌入和索引构建
                build_stats = self._build_index_for_chunks(version, chunks_data)
                
                # 激活新索引
                self._activate_index_version(version.id)
                
                logger.info(f"集合 {collection_id} 索引构建完成，版本: {version.id}")
                
                return IndexBuildResult(True, version.id, stats=build_stats)
                
            except Exception as e:
                logger.error(f"构建索引失败: {e}")
                self._update_version_status(version.id, IndexStatus.FAILED, str(e))
                return IndexBuildResult(False, version.id, str(e))
                
        except Exception as e:
            logger.error(f"索引构建过程异常: {e}")
            return IndexBuildResult(False, error=str(e))
            
    def _has_active_index(self, collection_id: str) -> bool:
        """检查是否有活跃的索引"""
        active_version = self.db.query(IndexVersion).filter(
            IndexVersion.collection_id == collection_id,
            IndexVersion.status == IndexStatus.ACTIVE
        ).first()
        return active_version is not None
        
    def _get_active_version(self, collection_id: str) -> Optional[IndexVersion]:
        """获取活跃的索引版本"""
        return self.db.query(IndexVersion).filter(
            IndexVersion.collection_id == collection_id,
            IndexVersion.status == IndexStatus.ACTIVE
        ).first()
        
    def _create_index_version(self, collection: Collection) -> IndexVersion:
        """创建新的索引版本记录"""
        # 生成索引文件路径
        timestamp = int(time.time())
        index_filename = f"index_{timestamp}.faiss"
        index_path = self.base_index_path / collection.id / index_filename
        
        # 解析索引配置
        index_config = json.loads(collection.index_config) if collection.index_config else {}
        embedding_model = index_config.get("embedding_model", "bge-m3")
        
        # 创建版本记录
        version = IndexVersion(
            collection_id=collection.id,
            status=IndexStatus.BUILDING,
            faiss_path=str(index_path),
            meta_json=json.dumps({
                "embedding_model": embedding_model,
                "dimension": None,  # 将在构建时填充
                "index_type": self._determine_index_type(collection.id),
                "created_at": datetime.utcnow().isoformat(),
                "build_config": index_config
            }),
            created_at=datetime.utcnow()
        )
        
        self.db.add(version)
        self.db.commit()
        
        logger.info(f"创建索引版本: {version.id}, 路径: {index_path}")
        return version
        
    def _determine_index_type(self, collection_id: str) -> str:
        """根据集合大小确定索引类型"""
        # 统计chunk数量
        # 首先获取该集合下的所有文档ID
        doc_ids = self.db.query(Document.id).filter(
            Document.collection_id == collection_id
        ).subquery()
        
        chunk_count = self.db.query(Chunk).join(Chunk.parent).filter(
            Parent.doc_id.in_(self.db.query(doc_ids.c.id))
        ).count()
        
        if chunk_count < 1000:
            return "flat"
        else:
            return "hnsw"
            
    def _get_chunks_for_indexing(self, collection_id: str) -> List[Tuple[str, str]]:
        """获取需要索引的chunk数据
        
        Returns:
            List of (chunk_id, text) tuples
        """
        # 查询所有未嵌入的chunk
        # 首先获取该集合下的所有文档ID
        doc_ids = self.db.query(Document.id).filter(
            Document.collection_id == collection_id
        ).subquery()
        
        chunks = self.db.query(Chunk.id, Chunk.text).join(Chunk.parent).filter(
            Parent.doc_id.in_(self.db.query(doc_ids.c.id)),
            Chunk.embedding == None  # 没有嵌入记录的chunk
        ).all()
        
        return [(chunk.id, chunk.text) for chunk in chunks]
    
    def _get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Tuple[str, str]]:
        """根据chunk_ids获取chunk数据"""
        chunks = self.db.query(Chunk.id, Chunk.text).filter(
            Chunk.id.in_(chunk_ids)
        ).all()
        
        return [(chunk.id, chunk.text) for chunk in chunks]
    
    def _get_active_version(self, collection_id: str) -> Optional[IndexVersion]:
        """获取活跃的索引版本"""
        return self.db.query(IndexVersion).filter(
            IndexVersion.collection_id == collection_id,
            IndexVersion.status == IndexStatus.ACTIVE
        ).first()
        
    def _build_index_for_chunks(self, version: IndexVersion, 
                               chunks_data: List[Tuple[str, str]]) -> Dict[str, Any]:
        """为chunk构建索引
        
        Args:
            version: 索引版本
            chunks_data: chunk数据列表
            
        Returns:
            构建统计信息
        """
        start_time = time.time()
        
        # 提取文本
        chunk_ids = [chunk_id for chunk_id, _ in chunks_data]
        texts = [text for _, text in chunks_data]
        
        logger.info(f"开始嵌入 {len(texts)} 个文本块")
        
        # 获取嵌入模型并进行批量嵌入
        meta = json.loads(version.meta_json)
        embedding_model = meta["embedding_model"]
        
        # 使用模型管理器获取嵌入提供商
        embedding_provider = self.model_manager.get_provider(embedding_model, ModelType.EMBEDDING)
        if not embedding_provider:
            raise RuntimeError(f"无法获取嵌入模型: {embedding_model}")
            
        embeddings = embedding_provider.embed_texts(texts)
        embed_time = time.time() - start_time
        
        # 更新版本元数据中的维度信息
        meta["dimension"] = embeddings.shape[1]
        meta["vector_count"] = len(embeddings)
        version.meta_json = json.dumps(meta)
        
        # 创建向量存储实例
        collection_name = f"collection_{version.collection_id}"
        vector_store = VectorStoreFactory.create(
            self.vector_store_type,
            {
                "storage_path": str(self.base_index_path / version.collection_id),
                "dimension": embeddings.shape[1],
                "index_type": meta.get("index_type", "flat")
            }
        )
        
        # 创建集合（如果不存在）
        if not vector_store.collection_exists(collection_name):
            vector_store.create_collection(collection_name, embeddings.shape[1])
            
        logger.info(f"开始构建向量索引，向量数量: {len(embeddings)}")
        index_start_time = time.time()
        
        # 准备向量记录
        vector_records = []
        for i, chunk_id in enumerate(chunk_ids):
            # 创建嵌入记录
            embedding_record = Embedding(
                vector_id=chunk_id,  # 直接使用chunk_id作为vector_id
                chunk_id=chunk_id,
                collection_id=version.collection_id,
                model=embedding_model,
                created_at=datetime.utcnow()
            )
            self.db.add(embedding_record)
            
            # 创建向量记录
            vector_records.append(VectorRecord(
                id=chunk_id,
                vector=embeddings[i],
                metadata={"chunk_id": chunk_id, "collection_id": version.collection_id}
            ))
            
        # 提交嵌入记录
        self.db.commit()
        
        # 批量插入向量
        vector_store.insert_vectors(collection_name, vector_records)
        
        index_time = time.time() - index_start_time
        total_time = time.time() - start_time
        
        # 构建统计信息
        stats = {
            "total_chunks": len(chunks_data),
            "total_vectors": len(embeddings),
            "dimension": embeddings.shape[1],
            "embed_time_seconds": embed_time,
            "index_time_seconds": index_time,
            "total_time_seconds": total_time,
            "embedding_model": embedding_model,
            "vector_store_type": self.vector_store_type
        }
        
        # 更新版本统计信息
        meta["build_stats"] = stats
        version.meta_json = json.dumps(meta)
        self.db.commit()
        
        logger.info(f"索引构建完成，统计: {stats}")
        return stats
        
    def _activate_index_version(self, version: IndexVersion) -> None:
        """激活索引版本"""
        logger.info(f"激活索引版本 {version.id}")
        
        # 更新版本状态
        version.status = IndexStatus.ACTIVE
        version.activated_at = datetime.utcnow()
        
        # 将其他版本设为非活跃
        self.db.query(IndexVersion).filter(
            IndexVersion.collection_id == version.collection_id,
            IndexVersion.id != version.id
        ).update({"status": IndexStatus.INACTIVE})
        
        self.db.commit()
        logger.info(f"索引版本 {version.id} 已激活")
        
    def _update_version_status(self, version_id: int, status: IndexStatus, error: Optional[str] = None) -> None:
        """更新版本状态"""
        version = self.db.query(IndexVersion).filter(IndexVersion.id == version_id).first()
        if version:
            version.status = status
            if error:
                meta = json.loads(version.meta_json)
                meta["error"] = error
                version.meta_json = json.dumps(meta)
            self.db.commit()
            
    def add_chunks_to_index(self, collection_id: str, chunk_ids: List[str]) -> IndexBuildResult:
        """向现有索引添加新的chunk
        
        Args:
            collection_id: 集合ID
            chunk_ids: 新增的chunk ID列表
            
        Returns:
            构建结果
        """
        try:
            # 获取活跃索引版本
            active_version = self._get_active_version(collection_id)
            if not active_version:
                logger.warning(f"集合 {collection_id} 没有活跃索引，执行全量构建")
                return self.build_collection_index(collection_id)
                
            # 获取新chunk的文本
            chunks = self.db.query(Chunk.id, Chunk.text).filter(
                Chunk.id.in_(chunk_ids)
            ).all()
            
            if not chunks:
                return IndexBuildResult(True, active_version.id)
                
            # 获取嵌入模型并进行嵌入
            texts = [chunk.text for chunk in chunks]
            meta = json.loads(active_version.meta_json)
            embedding_model = meta["embedding_model"]
            
            embedding_provider = self.model_manager.get_provider(embedding_model, ModelType.EMBEDDING)
            if not embedding_provider:
                raise RuntimeError(f"无法获取嵌入模型: {embedding_model}")
                
            embeddings = embedding_provider.embed_texts(texts)
            
            # 创建嵌入记录
            for chunk in chunks:
                embedding_record = Embedding(
                    vector_id=chunk.id,  # 直接使用chunk_id作为vector_id
                    chunk_id=chunk.id,
                    collection_id=collection_id,
                    model=embedding_model,
                    created_at=datetime.utcnow()
                )
                self.db.add(embedding_record)
                
            self.db.commit()
            
            # 获取向量存储并添加向量
            collection_name = f"collection_{collection_id}"
            vector_store = VectorStoreFactory.create_store(
                store_type=self.vector_store_type,
                collection_name=collection_name,
                dimension=meta["dimension"],
                config={
                    "index_path": str(self.base_index_path / collection_id),
                    "index_type": meta.get("index_type", "flat")
                }
            )
            
            # 准备向量记录
            vector_records = []
            for i, chunk in enumerate(chunks):
                vector_records.append(VectorRecord(
                    id=chunk.id,
                    vector=embeddings[i],
                    metadata={"chunk_id": chunk.id, "collection_id": collection_id}
                ))
            
            # 插入向量
            vector_store.insert_vectors(collection_name, vector_records)
            
            logger.info(f"成功添加 {len(chunk_ids)} 个chunk到索引")
            return IndexBuildResult(True, active_version.id)
            
        except Exception as e:
            logger.error(f"添加chunk到索引失败: {e}")
            return IndexBuildResult(False, error=str(e))
            
    def remove_chunks_from_index(self, collection_id: str, chunk_ids: List[str]) -> IndexBuildResult:
        """从索引中移除chunk
        
        Args:
            collection_id: 集合ID
            chunk_ids: 要移除的chunk ID列表
            
        Returns:
            构建结果
        """
        try:
            # 获取活跃索引版本
            active_version = self._get_active_version(collection_id)
            if not active_version:
                logger.warning(f"集合 {collection_id} 没有活跃索引")
                return IndexBuildResult(False, error="没有活跃索引")
            
            # 获取要删除的嵌入记录
            embeddings = self.db.query(Embedding).filter(
                Embedding.chunk_id.in_(chunk_ids),
                Embedding.collection_id == collection_id
            ).all()
            
            if not embeddings:
                return IndexBuildResult(True)
                
            # 获取向量存储
            meta = json.loads(active_version.meta_json)
            collection_name = f"collection_{collection_id}"
            vector_store = VectorStoreFactory.create_store(
                store_type=self.vector_store_type,
                collection_name=collection_name,
                dimension=meta["dimension"],
                config={
                    "index_path": str(self.base_index_path / collection_id),
                    "index_type": meta.get("index_type", "flat")
                }
            )
            
            # 从向量存储中删除向量
            vector_ids = [emb.chunk_id for emb in embeddings]  # 使用chunk_id作为vector_id
            vector_store.delete_vectors(collection_name, vector_ids)
            
            # 删除嵌入记录
            self.db.query(Embedding).filter(
                Embedding.chunk_id.in_(chunk_ids),
                Embedding.collection_id == collection_id
            ).delete(synchronize_session=False)
            
            # 更新统计信息
            meta["vector_count"] -= len(embeddings)
            active_version.meta_json = json.dumps(meta)
            
            self.db.commit()
                
            logger.info(f"成功从索引中移除 {len(embeddings)} 个向量")
            return IndexBuildResult(True)
            
        except Exception as e:
            logger.error(f"从索引移除chunk失败: {e}")
            return IndexBuildResult(False, error=str(e))
            
    def get_index_stats(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """获取索引统计信息
        
        Args:
            collection_id: 集合ID
            
        Returns:
            索引统计信息
        """
        active_version = self._get_active_version(collection_id)
        if not active_version:
            return None
            
        # 获取FAISS存储统计
        store_stats = self.store_manager.get_store(collection_id, 0).get_stats()
        
        # 合并版本信息
        version_meta = json.loads(active_version.meta_json)
        
        return {
            "version_id": active_version.id,
            "status": active_version.status,
            "created_at": active_version.created_at.isoformat(),
            "activated_at": active_version.activated_at.isoformat() if active_version.activated_at else None,
            "faiss_path": active_version.faiss_path,
            "meta": version_meta,
            "store_stats": store_stats
        }
        
    def cleanup_old_versions(self, collection_id: str, keep_count: int = 3) -> int:
        """清理旧的索引版本
        
        Args:
            collection_id: 集合ID
            keep_count: 保留的版本数量
            
        Returns:
            清理的版本数量
        """
        # 获取所有非活跃版本，按创建时间倒序
        old_versions = self.db.query(IndexVersion).filter(
            IndexVersion.collection_id == collection_id,
            IndexVersion.status != IndexStatus.ACTIVE
        ).order_by(IndexVersion.created_at.desc()).offset(keep_count).all()
        
        cleaned_count = 0
        for version in old_versions:
            try:
                # 删除索引文件
                if os.path.exists(version.faiss_path):
                    os.remove(version.faiss_path)
                    
                # 删除数据库记录
                self.db.delete(version)
                cleaned_count += 1
                
            except Exception as e:
                logger.error(f"清理版本 {version.id} 失败: {e}")
                
        if cleaned_count > 0:
            self.db.commit()
            logger.info(f"清理了 {cleaned_count} 个旧版本")
            
        return cleaned_count


# 全局构建器实例
_index_builder: Optional[IndexBuilder] = None


def get_index_builder(db_session: Session, base_path: str = "/data/indices") -> IndexBuilder:
    """获取索引构建器实例"""
    return IndexBuilder(db_session, base_path)