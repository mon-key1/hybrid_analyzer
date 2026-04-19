"""Ranking evaluation metrics (NDCG, MAP, Precision@k, Recall@k)."""

from typing import List, Set, Dict
import numpy as np


class RankingMetrics:
    """Compute ranking metrics for prioritization evaluation."""

    @staticmethod
    def ndcg_at_k(ranked_items: List[str],
                  relevant_items: List[str],
                  k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain at k.

        Args:
            ranked_items: List of cycle IDs ranked by predicted anomaly score
            relevant_items: List of cycle IDs ranked by actual elimination order
                           (first eliminated = most relevant)
            k: Cutoff position

        Returns:
            NDCG@k score in [0, 1]
        """
        def dcg_at_k(ranked, relevant, k):
            """Compute DCG@k."""
            dcg = 0.0
            for i in range(min(k, len(ranked))):
                item = ranked[i]
                if item in relevant:
                    # Relevance score: position in ground truth (inverse)
                    relevance = len(relevant) - relevant.index(item)
                    dcg += relevance / np.log2(i + 2)  # +2 because i is 0-indexed
            return dcg

        dcg = dcg_at_k(ranked_items, relevant_items, k)
        idcg = dcg_at_k(relevant_items, relevant_items, k)  # Ideal ranking

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def map_at_k(ranked_items: List[str],
                relevant_items: Set[str],
                k: int = 10) -> float:
        """
        Mean Average Precision at k.

        Args:
            ranked_items: Predicted ranking
            relevant_items: Set of items that are relevant (eliminated cycles)
            k: Cutoff position

        Returns:
            MAP@k score in [0, 1]
        """
        if not relevant_items:
            return 0.0

        precisions = []
        num_hits = 0

        for i, item in enumerate(ranked_items[:k]):
            if item in relevant_items:
                num_hits += 1
                precisions.append(num_hits / (i + 1))

        return np.mean(precisions) if precisions else 0.0

    @staticmethod
    def precision_at_k(ranked_items: List[str],
                      relevant_items: Set[str],
                      k: int = 5) -> float:
        """
        Precision at k.

        Args:
            ranked_items: Predicted ranking
            relevant_items: Set of relevant items
            k: Cutoff position

        Returns:
            Precision@k score in [0, 1]
        """
        if k == 0:
            return 0.0

        top_k = ranked_items[:k]
        hits = len([item for item in top_k if item in relevant_items])

        return hits / k

    @staticmethod
    def recall_at_k(ranked_items: List[str],
                   relevant_items: Set[str],
                   k: int = 10) -> float:
        """
        Recall at k.

        Args:
            ranked_items: Predicted ranking
            relevant_items: Set of relevant items
            k: Cutoff position

        Returns:
            Recall@k score in [0, 1]
        """
        if not relevant_items:
            return 0.0

        top_k = ranked_items[:k]
        hits = len([item for item in top_k if item in relevant_items])

        return hits / len(relevant_items)

    @staticmethod
    def f1_at_k(ranked_items: List[str],
               relevant_items: Set[str],
               k: int = 10) -> float:
        """
        F1 score at k.

        Args:
            ranked_items: Predicted ranking
            relevant_items: Set of relevant items
            k: Cutoff position

        Returns:
            F1@k score in [0, 1]
        """
        precision = RankingMetrics.precision_at_k(ranked_items, relevant_items, k)
        recall = RankingMetrics.recall_at_k(ranked_items, relevant_items, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def mrr(ranked_items: List[str],
           relevant_items: Set[str]) -> float:
        """
        Mean Reciprocal Rank.

        Args:
            ranked_items: Predicted ranking
            relevant_items: Set of relevant items

        Returns:
            MRR score
        """
        for i, item in enumerate(ranked_items):
            if item in relevant_items:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def evaluate_ranking(ranked_items: List[str],
                        relevant_items: List[str],
                        k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Compute all ranking metrics.

        Args:
            ranked_items: Predicted ranking
            relevant_items: Ground truth relevant items (ordered by relevance)
            k_values: List of k values for @k metrics

        Returns:
            Dictionary with all metric scores
        """
        relevant_set = set(relevant_items)

        metrics = {}

        for k in k_values:
            metrics[f'NDCG@{k}'] = RankingMetrics.ndcg_at_k(
                ranked_items, relevant_items, k
            )
            metrics[f'MAP@{k}'] = RankingMetrics.map_at_k(
                ranked_items, relevant_set, k
            )
            metrics[f'Precision@{k}'] = RankingMetrics.precision_at_k(
                ranked_items, relevant_set, k
            )
            metrics[f'Recall@{k}'] = RankingMetrics.recall_at_k(
                ranked_items, relevant_set, k
            )
            metrics[f'F1@{k}'] = RankingMetrics.f1_at_k(
                ranked_items, relevant_set, k
            )

        metrics['MRR'] = RankingMetrics.mrr(ranked_items, relevant_set)

        return metrics

    @staticmethod
    def compare_rankings(method_rankings: Dict[str, List[str]],
                        relevant_items: List[str],
                        k: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple ranking methods.

        Args:
            method_rankings: Dictionary mapping method name to ranked items
            relevant_items: Ground truth relevant items
            k: Cutoff for metrics

        Returns:
            Dictionary mapping method name to metrics dictionary
        """
        results = {}

        for method_name, ranked_items in method_rankings.items():
            metrics = RankingMetrics.evaluate_ranking(
                ranked_items,
                relevant_items,
                k_values=[k]
            )
            results[method_name] = metrics

        return results
