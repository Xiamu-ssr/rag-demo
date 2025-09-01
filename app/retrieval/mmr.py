"""MMR (Maximal Marginal Relevance) 去冗余算法模块

实现功能：
1. 最大边际相关性算法
2. 多样性和相关性平衡
3. 批量MMR处理
4. 自定义相似度计算
5. 增量MMR选择
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple, Any, Dict

import numpy as np

from app.retrieval.vector_retriever import RetrievalResult
from app.utils.vector_utils import cosine_similarity, euclidean_distance

logger = logging.getLogger(__name__)


@dataclass
class MMRConfig:
    """MMR配置"""
    lambda_param: float = 0.5  # 多样性权重 (0-1, 0=只考虑多样性, 1=只考虑相关性)
    top_k: int = 10  # 最终返回的结果数量
    candidate_k: int = 50  # 候选结果数量
    similarity_threshold: float = 0.8  # 相似度阈值，超过此值认为冗余
    enable_incremental: bool = True  # 是否启用增量选择
    
    def __post_init__(self):
        """验证配置参数"""
        if not 0 <= self.lambda_param <= 1:
            raise ValueError("lambda_param must be between 0 and 1")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")


class MMRSelector:
    """MMR选择器"""
    
    def __init__(self, similarity_func: Optional[Callable] = None):
        """
        初始化MMR选择器
        
        Args:
            similarity_func: 相似度计算函数，默认使用余弦相似度
        """
        self.similarity_func = similarity_func or cosine_similarity
        
        # 统计信息
        self.stats = {
            "total_selections": 0,
            "avg_selection_time": 0.0,
            "avg_diversity_score": 0.0,
            "avg_relevance_score": 0.0
        }
        
    def select(self, query_vector: np.ndarray, 
              candidates: List[RetrievalResult],
              config: Optional[MMRConfig] = None) -> List[RetrievalResult]:
        """执行MMR选择
        
        Args:
            query_vector: 查询向量
            candidates: 候选结果列表
            config: MMR配置
            
        Returns:
            去冗余后的结果列表
        """
        if not candidates:
            return []
            
        config = config or MMRConfig()
        start_time = time.time()
        
        try:
            # 限制候选数量
            limited_candidates = candidates[:config.candidate_k]
            
            # 获取候选向量
            candidate_vectors = self._get_candidate_vectors(limited_candidates)
            if candidate_vectors is None:
                logger.warning("无法获取候选向量，返回原始结果")
                return limited_candidates[:config.top_k]
                
            # 执行MMR算法
            if config.enable_incremental:
                selected_indices = self._incremental_mmr(
                    query_vector, candidate_vectors, config
                )
            else:
                selected_indices = self._batch_mmr(
                    query_vector, candidate_vectors, config
                )
                
            # 构建结果
            selected_results = [limited_candidates[i] for i in selected_indices]
            
            # 更新统计
            selection_time = time.time() - start_time
            self._update_stats(selection_time, query_vector, candidate_vectors, selected_indices)
            
            logger.debug(f"MMR选择完成: 候选={len(limited_candidates)}, "
                        f"选中={len(selected_results)}, "
                        f"λ={config.lambda_param}, "
                        f"耗时={selection_time:.3f}s")
                        
            return selected_results
            
        except Exception as e:
            logger.error(f"MMR选择失败: {e}")
            return candidates[:config.top_k]
            
    def _get_candidate_vectors(self, candidates: List[RetrievalResult]) -> Optional[np.ndarray]:
        """获取候选结果的向量表示
        
        Args:
            candidates: 候选结果列表
            
        Returns:
            候选向量矩阵 (n_candidates, vector_dim) 或 None
        """
        try:
            # 这里需要从存储中获取向量，暂时使用文本嵌入作为替代
            from app.llm import get_model_manager, ModelType
            
            model_manager = get_model_manager()
            embedder = model_manager.get_provider(ModelType.EMBEDDING)
            texts = [candidate.text for candidate in candidates]
            
            # 批量嵌入
            vectors = embedder.embed_texts(texts)
            return vectors
            
        except Exception as e:
            logger.error(f"获取候选向量失败: {e}")
            return None
            
    def _incremental_mmr(self, query_vector: np.ndarray, 
                        candidate_vectors: np.ndarray,
                        config: MMRConfig) -> List[int]:
        """增量MMR选择
        
        Args:
            query_vector: 查询向量
            candidate_vectors: 候选向量矩阵
            config: MMR配置
            
        Returns:
            选中的候选索引列表
        """
        n_candidates = len(candidate_vectors)
        selected_indices = []
        remaining_indices = list(range(n_candidates))
        
        # 计算查询相关性分数
        query_similarities = np.array([
            self.similarity_func(query_vector, candidate_vectors[i])
            for i in range(n_candidates)
        ])
        
        # 选择第一个结果（相关性最高）
        if remaining_indices:
            first_idx = remaining_indices[np.argmax(query_similarities)]
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            
        # 增量选择剩余结果
        while len(selected_indices) < config.top_k and remaining_indices:
            best_idx = None
            best_score = -float('inf')
            
            for candidate_idx in remaining_indices:
                # 计算与查询的相关性
                relevance = query_similarities[candidate_idx]
                
                # 计算与已选结果的最大相似度（多样性）
                max_similarity = 0.0
                if selected_indices:
                    similarities = [
                        self.similarity_func(
                            candidate_vectors[candidate_idx],
                            candidate_vectors[selected_idx]
                        )
                        for selected_idx in selected_indices
                    ]
                    max_similarity = max(similarities)
                    
                # 计算MMR分数
                mmr_score = (
                    config.lambda_param * relevance - 
                    (1 - config.lambda_param) * max_similarity
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = candidate_idx
                    
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
                
        return selected_indices
        
    def _batch_mmr(self, query_vector: np.ndarray,
                  candidate_vectors: np.ndarray,
                  config: MMRConfig) -> List[int]:
        """批量MMR选择（矩阵化计算）
        
        Args:
            query_vector: 查询向量
            candidate_vectors: 候选向量矩阵
            config: MMR配置
            
        Returns:
            选中的候选索引列表
        """
        n_candidates = len(candidate_vectors)
        
        # 计算查询相关性矩阵
        query_similarities = np.array([
            self.similarity_func(query_vector, candidate_vectors[i])
            for i in range(n_candidates)
        ])
        
        # 计算候选间相似度矩阵
        similarity_matrix = np.zeros((n_candidates, n_candidates))
        for i in range(n_candidates):
            for j in range(i + 1, n_candidates):
                sim = self.similarity_func(candidate_vectors[i], candidate_vectors[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
                
        # 贪心选择
        selected_indices = []
        remaining_mask = np.ones(n_candidates, dtype=bool)
        
        for _ in range(min(config.top_k, n_candidates)):
            if not np.any(remaining_mask):
                break
                
            mmr_scores = np.full(n_candidates, -float('inf'))
            
            for i in range(n_candidates):
                if not remaining_mask[i]:
                    continue
                    
                relevance = query_similarities[i]
                
                # 计算与已选结果的最大相似度
                max_similarity = 0.0
                if selected_indices:
                    max_similarity = max(
                        similarity_matrix[i, j] for j in selected_indices
                    )
                    
                mmr_scores[i] = (
                    config.lambda_param * relevance -
                    (1 - config.lambda_param) * max_similarity
                )
                
            # 选择最佳候选
            best_idx = np.argmax(mmr_scores)
            if mmr_scores[best_idx] > -float('inf'):
                selected_indices.append(best_idx)
                remaining_mask[best_idx] = False
            else:
                break
                
        return selected_indices
        
    def select_with_threshold(self, query_vector: np.ndarray,
                             candidates: List[RetrievalResult],
                             config: Optional[MMRConfig] = None) -> List[RetrievalResult]:
        """基于相似度阈值的MMR选择
        
        Args:
            query_vector: 查询向量
            candidates: 候选结果列表
            config: MMR配置
            
        Returns:
            去冗余后的结果列表
        """
        if not candidates:
            return []
            
        config = config or MMRConfig()
        
        try:
            # 获取候选向量
            candidate_vectors = self._get_candidate_vectors(candidates)
            if candidate_vectors is None:
                return candidates[:config.top_k]
                
            # 基于阈值的快速去重
            selected_indices = []
            
            for i, candidate_vector in enumerate(candidate_vectors):
                # 检查与已选结果的相似度
                is_redundant = False
                
                for selected_idx in selected_indices:
                    similarity = self.similarity_func(
                        candidate_vector, 
                        candidate_vectors[selected_idx]
                    )
                    
                    if similarity > config.similarity_threshold:
                        is_redundant = True
                        break
                        
                if not is_redundant:
                    selected_indices.append(i)
                    
                    if len(selected_indices) >= config.top_k:
                        break
                        
            return [candidates[i] for i in selected_indices]
            
        except Exception as e:
            logger.error(f"阈值MMR选择失败: {e}")
            return candidates[:config.top_k]
            
    def batch_select(self, query_vectors: List[np.ndarray],
                    candidate_lists: List[List[RetrievalResult]],
                    config: Optional[MMRConfig] = None) -> List[List[RetrievalResult]]:
        """批量MMR选择
        
        Args:
            query_vectors: 查询向量列表
            candidate_lists: 候选结果列表的列表
            config: MMR配置
            
        Returns:
            每个查询的去冗余结果列表
        """
        if len(query_vectors) != len(candidate_lists):
            raise ValueError("查询向量数量与候选列表数量不匹配")
            
        config = config or MMRConfig()
        start_time = time.time()
        
        results = []
        for query_vector, candidates in zip(query_vectors, candidate_lists):
            selected = self.select(query_vector, candidates, config)
            results.append(selected)
            
        batch_time = time.time() - start_time
        logger.info(f"批量MMR选择完成: {len(query_vectors)} 个查询, 耗时 {batch_time:.3f}s")
        
        return results
        
    def _update_stats(self, selection_time: float, query_vector: np.ndarray,
                     candidate_vectors: np.ndarray, selected_indices: List[int]) -> None:
        """更新统计信息"""
        self.stats["total_selections"] += 1
        
        # 更新平均选择时间
        total_time = self.stats["avg_selection_time"] * (self.stats["total_selections"] - 1) + selection_time
        self.stats["avg_selection_time"] = total_time / self.stats["total_selections"]
        
        # 计算多样性分数（选中结果间的平均相似度）
        if len(selected_indices) > 1:
            similarities = []
            for i in range(len(selected_indices)):
                for j in range(i + 1, len(selected_indices)):
                    sim = self.similarity_func(
                        candidate_vectors[selected_indices[i]],
                        candidate_vectors[selected_indices[j]]
                    )
                    similarities.append(sim)
                    
            diversity_score = 1.0 - np.mean(similarities) if similarities else 1.0
        else:
            diversity_score = 1.0
            
        # 计算相关性分数（选中结果与查询的平均相似度）
        relevance_scores = [
            self.similarity_func(query_vector, candidate_vectors[i])
            for i in selected_indices
        ]
        relevance_score = np.mean(relevance_scores) if relevance_scores else 0.0
        
        # 更新平均分数
        total_selections = self.stats["total_selections"]
        
        self.stats["avg_diversity_score"] = (
            self.stats["avg_diversity_score"] * (total_selections - 1) + diversity_score
        ) / total_selections
        
        self.stats["avg_relevance_score"] = (
            self.stats["avg_relevance_score"] * (total_selections - 1) + relevance_score
        ) / total_selections
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
        
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total_selections": 0,
            "avg_selection_time": 0.0,
            "avg_diversity_score": 0.0,
            "avg_relevance_score": 0.0
        }
        
    def explain_selection(self, query_vector: np.ndarray,
                         candidates: List[RetrievalResult],
                         config: Optional[MMRConfig] = None) -> Dict[str, Any]:
        """解释MMR选择过程（用于调试）
        
        Args:
            query_vector: 查询向量
            candidates: 候选结果列表
            config: MMR配置
            
        Returns:
            选择过程的详细信息
        """
        config = config or MMRConfig()
        explanation = {
            "config": config.__dict__,
            "candidates_count": len(candidates),
            "steps": [],
            "selected_results": []
        }
        
        try:
            # 获取候选向量
            candidate_vectors = self._get_candidate_vectors(candidates)
            if candidate_vectors is None:
                explanation["error"] = "无法获取候选向量"
                return explanation
                
            # 计算相关性分数
            relevance_scores = [
                self.similarity_func(query_vector, candidate_vectors[i])
                for i in range(len(candidates))
            ]
            
            explanation["steps"].append({
                "step": "relevance_calculation",
                "relevance_scores": relevance_scores[:10]  # 只显示前10个
            })
            
            # 执行选择
            selected_indices = self._incremental_mmr(query_vector, candidate_vectors, config)
            
            # 计算选择过程的详细信息
            selection_details = []
            for i, idx in enumerate(selected_indices):
                detail = {
                    "rank": i + 1,
                    "candidate_index": idx,
                    "relevance_score": relevance_scores[idx],
                    "text_preview": candidates[idx].text[:100] + "..."
                }
                
                # 计算与之前选中结果的相似度
                if i > 0:
                    similarities = [
                        self.similarity_func(
                            candidate_vectors[idx],
                            candidate_vectors[selected_indices[j]]
                        )
                        for j in range(i)
                    ]
                    detail["max_similarity_to_selected"] = max(similarities)
                    detail["avg_similarity_to_selected"] = np.mean(similarities)
                    
                selection_details.append(detail)
                
            explanation["steps"].append({
                "step": "mmr_selection",
                "selection_details": selection_details
            })
            
            explanation["selected_results"] = [
                candidates[i].to_dict() for i in selected_indices
            ]
            
        except Exception as e:
            explanation["error"] = str(e)
            
        return explanation


# 全局MMR选择器实例
_mmr_selector: Optional[MMRSelector] = None


def get_mmr_selector(similarity_func: Optional[Callable] = None) -> MMRSelector:
    """获取MMR选择器实例
    
    Args:
        similarity_func: 相似度计算函数
        
    Returns:
        MMR选择器实例
    """
    global _mmr_selector
    if _mmr_selector is None:
        _mmr_selector = MMRSelector(similarity_func)
    return _mmr_selector


def set_mmr_selector(selector: MMRSelector) -> None:
    """设置全局MMR选择器实例
    
    Args:
        selector: MMR选择器实例
    """
    global _mmr_selector
    _mmr_selector = selector