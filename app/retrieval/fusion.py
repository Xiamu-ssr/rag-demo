"""多库融合算法模块

实现功能：
1. RRF (Reciprocal Rank Fusion) 算法
2. 加权融合算法
3. 多源结果合并
4. 分数标准化和归一化
5. 自定义融合策略
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum

import numpy as np

from app.retrieval.vector_retriever import RetrievalResult

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """融合方法枚举"""
    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"  # 加权求和
    MAX_SCORE = "max_score"  # 最大分数
    MIN_SCORE = "min_score"  # 最小分数
    AVERAGE = "average"  # 平均分数
    HARMONIC_MEAN = "harmonic_mean"  # 调和平均
    GEOMETRIC_MEAN = "geometric_mean"  # 几何平均


@dataclass
class FusionConfig:
    """融合配置"""
    method: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60  # RRF参数k
    weights: Optional[Dict[str, float]] = None  # 各源权重
    normalize_scores: bool = True  # 是否标准化分数
    top_k: int = 10  # 最终返回结果数量
    score_threshold: float = 0.0  # 分数阈值
    enable_deduplication: bool = True  # 是否去重
    
    def __post_init__(self):
        """验证配置参数"""
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be between 0 and 1")


@dataclass
class SourceResults:
    """单个源的检索结果"""
    source_id: str
    results: List[RetrievalResult]
    weight: float = 1.0
    
    def __post_init__(self):
        """验证参数"""
        if self.weight < 0:
            raise ValueError("weight must be non-negative")


class ResultFusion:
    """结果融合器"""
    
    def __init__(self):
        """
        初始化结果融合器
        """
        # 统计信息
        self.stats = {
            "total_fusions": 0,
            "avg_fusion_time": 0.0,
            "avg_sources_per_fusion": 0.0,
            "avg_results_per_source": 0.0
        }
        
    def fuse(self, source_results: List[SourceResults],
            config: Optional[FusionConfig] = None) -> List[RetrievalResult]:
        """融合多个源的检索结果
        
        Args:
            source_results: 各源的检索结果
            config: 融合配置
            
        Returns:
            融合后的结果列表
        """
        if not source_results:
            return []
            
        config = config or FusionConfig()
        
        try:
            import time
            start_time = time.time()
            
            # 预处理：标准化分数
            if config.normalize_scores:
                source_results = self._normalize_source_scores(source_results)
                
            # 应用权重
            if config.weights:
                source_results = self._apply_weights(source_results, config.weights)
                
            # 执行融合
            if config.method == FusionMethod.RRF:
                fused_results = self._rrf_fusion(source_results, config)
            elif config.method == FusionMethod.WEIGHTED_SUM:
                fused_results = self._weighted_sum_fusion(source_results, config)
            elif config.method == FusionMethod.MAX_SCORE:
                fused_results = self._max_score_fusion(source_results, config)
            elif config.method == FusionMethod.MIN_SCORE:
                fused_results = self._min_score_fusion(source_results, config)
            elif config.method == FusionMethod.AVERAGE:
                fused_results = self._average_fusion(source_results, config)
            elif config.method == FusionMethod.HARMONIC_MEAN:
                fused_results = self._harmonic_mean_fusion(source_results, config)
            elif config.method == FusionMethod.GEOMETRIC_MEAN:
                fused_results = self._geometric_mean_fusion(source_results, config)
            else:
                raise ValueError(f"不支持的融合方法: {config.method}")
                
            # 去重
            if config.enable_deduplication:
                fused_results = self._deduplicate_results(fused_results)
                
            # 过滤和排序
            final_results = self._filter_and_sort(fused_results, config)
            
            # 更新统计
            fusion_time = time.time() - start_time
            self._update_stats(fusion_time, source_results)
            
            logger.debug(f"融合完成: 方法={config.method.value}, "
                        f"源数={len(source_results)}, "
                        f"结果数={len(final_results)}, "
                        f"耗时={fusion_time:.3f}s")
                        
            return final_results
            
        except Exception as e:
            logger.error(f"结果融合失败: {e}")
            # 返回第一个源的结果作为fallback
            if source_results and source_results[0].results:
                return source_results[0].results[:config.top_k]
            return []
            
    def _normalize_source_scores(self, source_results: List[SourceResults]) -> List[SourceResults]:
        """标准化各源的分数
        
        Args:
            source_results: 原始源结果
            
        Returns:
            标准化后的源结果
        """
        normalized_results = []
        
        for source_result in source_results:
            if not source_result.results:
                normalized_results.append(source_result)
                continue
                
            scores = [r.score for r in source_result.results]
            min_score = min(scores)
            max_score = max(scores)
            
            # 避免除零
            if max_score == min_score:
                normalized_scores = [1.0] * len(scores)
            else:
                normalized_scores = [
                    (score - min_score) / (max_score - min_score)
                    for score in scores
                ]
                
            # 创建新的结果列表
            normalized_result_list = []
            for result, norm_score in zip(source_result.results, normalized_scores):
                new_result = RetrievalResult(
                    chunk_id=result.chunk_id,
                    vector_id=result.vector_id,
                    score=norm_score,
                    text=result.text,
                    metadata=result.metadata.copy(),
                    collection_id=result.collection_id
                )
                normalized_result_list.append(new_result)
                
            normalized_source = SourceResults(
                source_id=source_result.source_id,
                results=normalized_result_list,
                weight=source_result.weight
            )
            normalized_results.append(normalized_source)
            
        return normalized_results
        
    def _apply_weights(self, source_results: List[SourceResults],
                      weights: Dict[str, float]) -> List[SourceResults]:
        """应用源权重
        
        Args:
            source_results: 源结果列表
            weights: 权重字典
            
        Returns:
            应用权重后的源结果
        """
        weighted_results = []
        
        for source_result in source_results:
            weight = weights.get(source_result.source_id, source_result.weight)
            
            weighted_source = SourceResults(
                source_id=source_result.source_id,
                results=source_result.results,
                weight=weight
            )
            weighted_results.append(weighted_source)
            
        return weighted_results
        
    def _rrf_fusion(self, source_results: List[SourceResults],
                   config: FusionConfig) -> List[RetrievalResult]:
        """RRF融合算法
        
        Args:
            source_results: 源结果列表
            config: 融合配置
            
        Returns:
            融合后的结果列表
        """
        # 收集所有唯一的chunk
        chunk_scores = {}
        chunk_results = {}
        
        for source_result in source_results:
            for rank, result in enumerate(source_result.results):
                chunk_id = result.chunk_id
                
                # 计算RRF分数
                rrf_score = source_result.weight / (config.rrf_k + rank + 1)
                
                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id] += rrf_score
                else:
                    chunk_scores[chunk_id] = rrf_score
                    chunk_results[chunk_id] = result
                    
        # 创建融合结果
        fused_results = []
        for chunk_id, score in chunk_scores.items():
            result = chunk_results[chunk_id]
            fused_result = RetrievalResult(
                chunk_id=result.chunk_id,
                vector_id=result.vector_id,
                score=score,
                text=result.text,
                metadata=result.metadata.copy(),
                collection_id=result.collection_id
            )
            fused_results.append(fused_result)
            
        return fused_results
        
    def _weighted_sum_fusion(self, source_results: List[SourceResults],
                           config: FusionConfig) -> List[RetrievalResult]:
        """加权求和融合
        
        Args:
            source_results: 源结果列表
            config: 融合配置
            
        Returns:
            融合后的结果列表
        """
        chunk_scores = {}
        chunk_results = {}
        chunk_weights = {}
        
        for source_result in source_results:
            for result in source_result.results:
                chunk_id = result.chunk_id
                weighted_score = result.score * source_result.weight
                
                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id] += weighted_score
                    chunk_weights[chunk_id] += source_result.weight
                else:
                    chunk_scores[chunk_id] = weighted_score
                    chunk_weights[chunk_id] = source_result.weight
                    chunk_results[chunk_id] = result
                    
        # 计算加权平均分数
        fused_results = []
        for chunk_id, total_score in chunk_scores.items():
            total_weight = chunk_weights[chunk_id]
            avg_score = total_score / total_weight if total_weight > 0 else 0
            
            result = chunk_results[chunk_id]
            fused_result = RetrievalResult(
                chunk_id=result.chunk_id,
                vector_id=result.vector_id,
                score=avg_score,
                text=result.text,
                metadata=result.metadata.copy(),
                collection_id=result.collection_id
            )
            fused_results.append(fused_result)
            
        return fused_results
        
    def _max_score_fusion(self, source_results: List[SourceResults],
                         config: FusionConfig) -> List[RetrievalResult]:
        """最大分数融合"""
        return self._score_aggregation_fusion(source_results, max)
        
    def _min_score_fusion(self, source_results: List[SourceResults],
                         config: FusionConfig) -> List[RetrievalResult]:
        """最小分数融合"""
        return self._score_aggregation_fusion(source_results, min)
        
    def _average_fusion(self, source_results: List[SourceResults],
                       config: FusionConfig) -> List[RetrievalResult]:
        """平均分数融合"""
        return self._score_aggregation_fusion(source_results, lambda scores: sum(scores) / len(scores))
        
    def _harmonic_mean_fusion(self, source_results: List[SourceResults],
                             config: FusionConfig) -> List[RetrievalResult]:
        """调和平均融合"""
        def harmonic_mean(scores):
            if any(s <= 0 for s in scores):
                return 0.0
            return len(scores) / sum(1/s for s in scores)
            
        return self._score_aggregation_fusion(source_results, harmonic_mean)
        
    def _geometric_mean_fusion(self, source_results: List[SourceResults],
                              config: FusionConfig) -> List[RetrievalResult]:
        """几何平均融合"""
        def geometric_mean(scores):
            if any(s <= 0 for s in scores):
                return 0.0
            product = 1.0
            for s in scores:
                product *= s
            return product ** (1.0 / len(scores))
            
        return self._score_aggregation_fusion(source_results, geometric_mean)
        
    def _score_aggregation_fusion(self, source_results: List[SourceResults],
                                 aggregation_func: Callable) -> List[RetrievalResult]:
        """通用分数聚合融合
        
        Args:
            source_results: 源结果列表
            aggregation_func: 聚合函数
            
        Returns:
            融合后的结果列表
        """
        chunk_scores = {}
        chunk_results = {}
        
        for source_result in source_results:
            for result in source_result.results:
                chunk_id = result.chunk_id
                weighted_score = result.score * source_result.weight
                
                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id].append(weighted_score)
                else:
                    chunk_scores[chunk_id] = [weighted_score]
                    chunk_results[chunk_id] = result
                    
        # 应用聚合函数
        fused_results = []
        for chunk_id, scores in chunk_scores.items():
            aggregated_score = aggregation_func(scores)
            
            result = chunk_results[chunk_id]
            fused_result = RetrievalResult(
                chunk_id=result.chunk_id,
                vector_id=result.vector_id,
                score=aggregated_score,
                text=result.text,
                metadata=result.metadata.copy(),
                collection_id=result.collection_id
            )
            fused_results.append(fused_result)
            
        return fused_results
        
    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """去重结果（基于chunk_id）
        
        Args:
            results: 原始结果列表
            
        Returns:
            去重后的结果列表
        """
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                unique_results.append(result)
                
        return unique_results
        
    def _filter_and_sort(self, results: List[RetrievalResult],
                        config: FusionConfig) -> List[RetrievalResult]:
        """过滤和排序结果
        
        Args:
            results: 原始结果列表
            config: 融合配置
            
        Returns:
            过滤排序后的结果列表
        """
        # 过滤低分结果
        filtered_results = [
            result for result in results
            if result.score >= config.score_threshold
        ]
        
        # 按分数降序排序
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        # 限制结果数量
        return filtered_results[:config.top_k]
        
    def _update_stats(self, fusion_time: float, source_results: List[SourceResults]) -> None:
        """更新统计信息"""
        self.stats["total_fusions"] += 1
        
        # 更新平均融合时间
        total_time = self.stats["avg_fusion_time"] * (self.stats["total_fusions"] - 1) + fusion_time
        self.stats["avg_fusion_time"] = total_time / self.stats["total_fusions"]
        
        # 更新平均源数量
        total_sources = self.stats["avg_sources_per_fusion"] * (self.stats["total_fusions"] - 1) + len(source_results)
        self.stats["avg_sources_per_fusion"] = total_sources / self.stats["total_fusions"]
        
        # 更新平均每源结果数
        if source_results:
            avg_results = sum(len(sr.results) for sr in source_results) / len(source_results)
            total_avg_results = self.stats["avg_results_per_source"] * (self.stats["total_fusions"] - 1) + avg_results
            self.stats["avg_results_per_source"] = total_avg_results / self.stats["total_fusions"]
            
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
        
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total_fusions": 0,
            "avg_fusion_time": 0.0,
            "avg_sources_per_fusion": 0.0,
            "avg_results_per_source": 0.0
        }
        
    def explain_fusion(self, source_results: List[SourceResults],
                      config: Optional[FusionConfig] = None) -> Dict[str, Any]:
        """解释融合过程（用于调试）
        
        Args:
            source_results: 源结果列表
            config: 融合配置
            
        Returns:
            融合过程的详细信息
        """
        config = config or FusionConfig()
        explanation = {
            "config": {
                "method": config.method.value,
                "rrf_k": config.rrf_k,
                "weights": config.weights,
                "normalize_scores": config.normalize_scores,
                "top_k": config.top_k
            },
            "sources": [],
            "fusion_details": {},
            "final_results": []
        }
        
        try:
            # 源信息
            for source_result in source_results:
                source_info = {
                    "source_id": source_result.source_id,
                    "weight": source_result.weight,
                    "result_count": len(source_result.results),
                    "score_range": {
                        "min": min(r.score for r in source_result.results) if source_result.results else 0,
                        "max": max(r.score for r in source_result.results) if source_result.results else 0
                    },
                    "top_results": [
                        {
                            "chunk_id": r.chunk_id,
                            "score": r.score,
                            "text_preview": r.text[:50] + "..."
                        }
                        for r in source_result.results[:3]
                    ]
                }
                explanation["sources"].append(source_info)
                
            # 执行融合并记录详情
            if config.method == FusionMethod.RRF:
                explanation["fusion_details"]["method_description"] = "Reciprocal Rank Fusion"
                explanation["fusion_details"]["formula"] = f"score = weight / (k + rank + 1), k={config.rrf_k}"
            elif config.method == FusionMethod.WEIGHTED_SUM:
                explanation["fusion_details"]["method_description"] = "Weighted Sum Fusion"
                explanation["fusion_details"]["formula"] = "score = sum(weight * score) / sum(weight)"
                
            # 执行实际融合
            fused_results = self.fuse(source_results, config)
            
            explanation["final_results"] = [
                {
                    "rank": i + 1,
                    "chunk_id": result.chunk_id,
                    "final_score": result.score,
                    "text_preview": result.text[:100] + "..."
                }
                for i, result in enumerate(fused_results)
            ]
            
        except Exception as e:
            explanation["error"] = str(e)
            
        return explanation


# 全局融合器实例
_result_fusion: Optional[ResultFusion] = None


def get_result_fusion() -> ResultFusion:
    """获取结果融合器实例
    
    Returns:
        结果融合器实例
    """
    global _result_fusion
    if _result_fusion is None:
        _result_fusion = ResultFusion()
    return _result_fusion


def set_result_fusion(fusion: ResultFusion) -> None:
    """设置全局结果融合器实例
    
    Args:
        fusion: 结果融合器实例
    """
    global _result_fusion
    _result_fusion = fusion