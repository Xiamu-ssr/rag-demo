"""向量处理工具函数模块

实现功能：
1. 向量归一化（L1、L2、Max归一化）
2. 相似度计算（余弦、欧几里得、曼哈顿、点积）
3. 向量距离计算
4. 向量维度检查和验证
5. 批量向量处理
6. 向量统计分析
"""

import logging
import math
from typing import List, Tuple, Union, Optional, Dict, Any
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class NormalizationType(Enum):
    """归一化类型枚举"""
    L1 = "l1"  # L1归一化（曼哈顿范数）
    L2 = "l2"  # L2归一化（欧几里得范数）
    MAX = "max"  # 最大值归一化
    MIN_MAX = "min_max"  # 最小-最大归一化
    Z_SCORE = "z_score"  # Z-score标准化
    UNIT = "unit"  # 单位向量归一化


class SimilarityType(Enum):
    """相似度类型枚举"""
    COSINE = "cosine"  # 余弦相似度
    DOT_PRODUCT = "dot_product"  # 点积相似度
    EUCLIDEAN = "euclidean"  # 欧几里得距离（转换为相似度）
    MANHATTAN = "manhattan"  # 曼哈顿距离（转换为相似度）
    JACCARD = "jaccard"  # 雅卡德相似度
    PEARSON = "pearson"  # 皮尔逊相关系数


class VectorProcessor:
    """向量处理器"""
    
    def __init__(self):
        """
        初始化向量处理器
        """
        # 统计信息
        self.stats = {
            "total_normalizations": 0,
            "total_similarity_calculations": 0,
            "avg_vector_dimension": 0.0,
            "processing_errors": 0
        }
        
    def normalize_vector(self, vector: Union[List[float], np.ndarray],
                        norm_type: NormalizationType = NormalizationType.L2) -> np.ndarray:
        """归一化单个向量
        
        Args:
            vector: 输入向量
            norm_type: 归一化类型
            
        Returns:
            归一化后的向量
        """
        try:
            vector = np.array(vector, dtype=np.float32)
            
            if vector.size == 0:
                raise ValueError("向量不能为空")
                
            if norm_type == NormalizationType.L1:
                norm = np.sum(np.abs(vector))
                if norm == 0:
                    return vector
                return vector / norm
                
            elif norm_type == NormalizationType.L2:
                norm = np.linalg.norm(vector)
                if norm == 0:
                    return vector
                return vector / norm
                
            elif norm_type == NormalizationType.MAX:
                max_val = np.max(np.abs(vector))
                if max_val == 0:
                    return vector
                return vector / max_val
                
            elif norm_type == NormalizationType.MIN_MAX:
                min_val = np.min(vector)
                max_val = np.max(vector)
                if max_val == min_val:
                    return np.zeros_like(vector)
                return (vector - min_val) / (max_val - min_val)
                
            elif norm_type == NormalizationType.Z_SCORE:
                mean = np.mean(vector)
                std = np.std(vector)
                if std == 0:
                    return np.zeros_like(vector)
                return (vector - mean) / std
                
            elif norm_type == NormalizationType.UNIT:
                return self.normalize_vector(vector, NormalizationType.L2)
                
            else:
                raise ValueError(f"不支持的归一化类型: {norm_type}")
                
        except Exception as e:
            logger.error(f"向量归一化失败: {e}")
            self.stats["processing_errors"] += 1
            return np.array(vector, dtype=np.float32)
        finally:
            self.stats["total_normalizations"] += 1
            
    def normalize_vectors(self, vectors: Union[List[List[float]], np.ndarray],
                         norm_type: NormalizationType = NormalizationType.L2) -> np.ndarray:
        """批量归一化向量
        
        Args:
            vectors: 输入向量列表或矩阵
            norm_type: 归一化类型
            
        Returns:
            归一化后的向量矩阵
        """
        try:
            vectors = np.array(vectors, dtype=np.float32)
            
            if vectors.ndim == 1:
                return self.normalize_vector(vectors, norm_type)
                
            if vectors.ndim != 2:
                raise ValueError("向量矩阵必须是2维的")
                
            normalized_vectors = []
            for vector in vectors:
                normalized_vector = self.normalize_vector(vector, norm_type)
                normalized_vectors.append(normalized_vector)
                
            return np.array(normalized_vectors, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"批量向量归一化失败: {e}")
            self.stats["processing_errors"] += 1
            return np.array(vectors, dtype=np.float32)
            
    def calculate_similarity(self, vector1: Union[List[float], np.ndarray],
                           vector2: Union[List[float], np.ndarray],
                           similarity_type: SimilarityType = SimilarityType.COSINE) -> float:
        """计算两个向量的相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            similarity_type: 相似度类型
            
        Returns:
            相似度分数
        """
        try:
            vector1 = np.array(vector1, dtype=np.float32)
            vector2 = np.array(vector2, dtype=np.float32)
            
            if vector1.shape != vector2.shape:
                raise ValueError("向量维度必须相同")
                
            if vector1.size == 0:
                raise ValueError("向量不能为空")
                
            if similarity_type == SimilarityType.COSINE:
                return self._cosine_similarity(vector1, vector2)
                
            elif similarity_type == SimilarityType.DOT_PRODUCT:
                return float(np.dot(vector1, vector2))
                
            elif similarity_type == SimilarityType.EUCLIDEAN:
                distance = np.linalg.norm(vector1 - vector2)
                # 转换为相似度（距离越小，相似度越高）
                return 1.0 / (1.0 + distance)
                
            elif similarity_type == SimilarityType.MANHATTAN:
                distance = np.sum(np.abs(vector1 - vector2))
                # 转换为相似度
                return 1.0 / (1.0 + distance)
                
            elif similarity_type == SimilarityType.JACCARD:
                return self._jaccard_similarity(vector1, vector2)
                
            elif similarity_type == SimilarityType.PEARSON:
                return self._pearson_correlation(vector1, vector2)
                
            else:
                raise ValueError(f"不支持的相似度类型: {similarity_type}")
                
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            self.stats["processing_errors"] += 1
            return 0.0
        finally:
            self.stats["total_similarity_calculations"] += 1
            
    def batch_similarity(self, query_vector: Union[List[float], np.ndarray],
                        vectors: Union[List[List[float]], np.ndarray],
                        similarity_type: SimilarityType = SimilarityType.COSINE) -> List[float]:
        """批量计算查询向量与多个向量的相似度
        
        Args:
            query_vector: 查询向量
            vectors: 目标向量列表
            similarity_type: 相似度类型
            
        Returns:
            相似度分数列表
        """
        try:
            query_vector = np.array(query_vector, dtype=np.float32)
            vectors = np.array(vectors, dtype=np.float32)
            
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
                
            if vectors.ndim != 2:
                raise ValueError("向量矩阵必须是2维的")
                
            similarities = []
            for vector in vectors:
                similarity = self.calculate_similarity(query_vector, vector, similarity_type)
                similarities.append(similarity)
                
            return similarities
            
        except Exception as e:
            logger.error(f"批量相似度计算失败: {e}")
            self.stats["processing_errors"] += 1
            return [0.0] * len(vectors)
            
    def _cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """计算余弦相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            余弦相似度
        """
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(vector1, vector2) / (norm1 * norm2))
        
    def _jaccard_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """计算雅卡德相似度（适用于二值向量）
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            雅卡德相似度
        """
        # 将向量转换为二值
        binary1 = (vector1 > 0).astype(int)
        binary2 = (vector2 > 0).astype(int)
        
        intersection = np.sum(binary1 & binary2)
        union = np.sum(binary1 | binary2)
        
        if union == 0:
            return 0.0
            
        return float(intersection / union)
        
    def _pearson_correlation(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """计算皮尔逊相关系数
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            皮尔逊相关系数
        """
        if len(vector1) < 2:
            return 0.0
            
        mean1 = np.mean(vector1)
        mean2 = np.mean(vector2)
        
        numerator = np.sum((vector1 - mean1) * (vector2 - mean2))
        denominator = np.sqrt(np.sum((vector1 - mean1) ** 2) * np.sum((vector2 - mean2) ** 2))
        
        if denominator == 0:
            return 0.0
            
        return float(numerator / denominator)
        
    def calculate_distance(self, vector1: Union[List[float], np.ndarray],
                         vector2: Union[List[float], np.ndarray],
                         distance_type: str = "euclidean") -> float:
        """计算两个向量的距离
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            distance_type: 距离类型（euclidean, manhattan, chebyshev）
            
        Returns:
            距离值
        """
        try:
            vector1 = np.array(vector1, dtype=np.float32)
            vector2 = np.array(vector2, dtype=np.float32)
            
            if vector1.shape != vector2.shape:
                raise ValueError("向量维度必须相同")
                
            if distance_type == "euclidean":
                return float(np.linalg.norm(vector1 - vector2))
            elif distance_type == "manhattan":
                return float(np.sum(np.abs(vector1 - vector2)))
            elif distance_type == "chebyshev":
                return float(np.max(np.abs(vector1 - vector2)))
            else:
                raise ValueError(f"不支持的距离类型: {distance_type}")
                
        except Exception as e:
            logger.error(f"距离计算失败: {e}")
            return float('inf')
            
    def validate_vector_dimensions(self, vectors: Union[List[List[float]], np.ndarray],
                                 expected_dim: Optional[int] = None) -> Tuple[bool, str]:
        """验证向量维度
        
        Args:
            vectors: 向量列表或矩阵
            expected_dim: 期望的维度
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            vectors = np.array(vectors, dtype=np.float32)
            
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
                
            if vectors.ndim != 2:
                return False, "向量矩阵必须是2维的"
                
            if vectors.size == 0:
                return False, "向量不能为空"
                
            # 检查所有向量维度是否一致
            first_dim = vectors.shape[1]
            if not all(len(v) == first_dim for v in vectors):
                return False, "所有向量维度必须一致"
                
            # 检查期望维度
            if expected_dim is not None and first_dim != expected_dim:
                return False, f"向量维度{first_dim}与期望维度{expected_dim}不匹配"
                
            # 检查向量是否包含无效值
            if np.any(np.isnan(vectors)) or np.any(np.isinf(vectors)):
                return False, "向量包含NaN或无穷大值"
                
            return True, "向量维度验证通过"
            
        except Exception as e:
            return False, f"向量维度验证失败: {e}"
            
    def get_vector_stats(self, vectors: Union[List[List[float]], np.ndarray]) -> Dict[str, Any]:
        """获取向量统计信息
        
        Args:
            vectors: 向量列表或矩阵
            
        Returns:
            统计信息字典
        """
        try:
            vectors = np.array(vectors, dtype=np.float32)
            
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
                
            stats = {
                "count": vectors.shape[0],
                "dimension": vectors.shape[1],
                "mean": np.mean(vectors, axis=0).tolist(),
                "std": np.std(vectors, axis=0).tolist(),
                "min": np.min(vectors, axis=0).tolist(),
                "max": np.max(vectors, axis=0).tolist(),
                "norm_l1": np.mean(np.sum(np.abs(vectors), axis=1)),
                "norm_l2": np.mean(np.linalg.norm(vectors, axis=1)),
                "sparsity": np.mean(vectors == 0),
                "has_nan": bool(np.any(np.isnan(vectors))),
                "has_inf": bool(np.any(np.isinf(vectors)))
            }
            
            # 更新处理器统计
            if self.stats["avg_vector_dimension"] == 0:
                self.stats["avg_vector_dimension"] = vectors.shape[1]
            else:
                # 计算移动平均
                self.stats["avg_vector_dimension"] = (
                    self.stats["avg_vector_dimension"] * 0.9 + vectors.shape[1] * 0.1
                )
                
            return stats
            
        except Exception as e:
            logger.error(f"向量统计计算失败: {e}")
            return {}
            
    def find_most_similar(self, query_vector: Union[List[float], np.ndarray],
                         vectors: Union[List[List[float]], np.ndarray],
                         top_k: int = 5,
                         similarity_type: SimilarityType = SimilarityType.COSINE) -> List[Tuple[int, float]]:
        """找到最相似的向量
        
        Args:
            query_vector: 查询向量
            vectors: 候选向量列表
            top_k: 返回前k个最相似的
            similarity_type: 相似度类型
            
        Returns:
            (索引, 相似度分数)的列表，按相似度降序排列
        """
        try:
            similarities = self.batch_similarity(query_vector, vectors, similarity_type)
            
            # 创建(索引, 相似度)对并排序
            indexed_similarities = [(i, sim) for i, sim in enumerate(similarities)]
            indexed_similarities.sort(key=lambda x: x[1], reverse=True)
            
            return indexed_similarities[:top_k]
            
        except Exception as e:
            logger.error(f"最相似向量查找失败: {e}")
            return []
            
    def cluster_vectors(self, vectors: Union[List[List[float]], np.ndarray],
                       n_clusters: int = 5,
                       similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """简单的向量聚类（基于相似度阈值）
        
        Args:
            vectors: 向量列表
            n_clusters: 期望的聚类数量
            similarity_threshold: 相似度阈值
            
        Returns:
            聚类结果字典
        """
        try:
            vectors = np.array(vectors, dtype=np.float32)
            
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
                
            clusters = []
            assigned = set()
            
            for i, vector in enumerate(vectors):
                if i in assigned:
                    continue
                    
                # 创建新聚类
                cluster = [i]
                assigned.add(i)
                
                # 找到相似的向量
                for j, other_vector in enumerate(vectors):
                    if j in assigned or i == j:
                        continue
                        
                    similarity = self.calculate_similarity(vector, other_vector)
                    if similarity >= similarity_threshold:
                        cluster.append(j)
                        assigned.add(j)
                        
                clusters.append(cluster)
                
                # 限制聚类数量
                if len(clusters) >= n_clusters:
                    break
                    
            # 处理未分配的向量
            unassigned = [i for i in range(len(vectors)) if i not in assigned]
            if unassigned:
                clusters.append(unassigned)
                
            return {
                "clusters": clusters,
                "n_clusters": len(clusters),
                "cluster_sizes": [len(cluster) for cluster in clusters],
                "assigned_count": len(assigned),
                "unassigned_count": len(unassigned)
            }
            
        except Exception as e:
            logger.error(f"向量聚类失败: {e}")
            return {"clusters": [], "n_clusters": 0}
            
    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
        
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total_normalizations": 0,
            "total_similarity_calculations": 0,
            "avg_vector_dimension": 0.0,
            "processing_errors": 0
        }


# 全局向量处理器实例
_vector_processor: Optional[VectorProcessor] = None


def get_vector_processor() -> VectorProcessor:
    """获取向量处理器实例
    
    Returns:
        向量处理器实例
    """
    global _vector_processor
    if _vector_processor is None:
        _vector_processor = VectorProcessor()
    return _vector_processor


def set_vector_processor(processor: VectorProcessor) -> None:
    """设置全局向量处理器实例
    
    Args:
        processor: 向量处理器实例
    """
    global _vector_processor
    _vector_processor = processor


# 便捷函数
def normalize_vector(vector: Union[List[float], np.ndarray],
                    norm_type: NormalizationType = NormalizationType.L2) -> np.ndarray:
    """归一化向量的便捷函数"""
    return get_vector_processor().normalize_vector(vector, norm_type)


def cosine_similarity(vector1: Union[List[float], np.ndarray],
                     vector2: Union[List[float], np.ndarray]) -> float:
    """计算余弦相似度的便捷函数"""
    return get_vector_processor().calculate_similarity(vector1, vector2, SimilarityType.COSINE)


def euclidean_distance(vector1: Union[List[float], np.ndarray],
                      vector2: Union[List[float], np.ndarray]) -> float:
    """计算欧几里得距离的便捷函数"""
    return get_vector_processor().calculate_distance(vector1, vector2, "euclidean")