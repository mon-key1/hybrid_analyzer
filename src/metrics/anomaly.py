"""Anomaly score computation for cyclic dependencies."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from ..semantic.similarity import SimilarityCalculator


class AnomalyScorer:
    """Compute anomaly scores for cycles based on semantic, structural, and dynamic analysis."""

    def __init__(self,
                 semantic_weight: float = 0.4,
                 structural_weight: float = 0.4,
                 dynamic_weight: float = 0.2):
        """
        Initialize anomaly scorer with component weights.

        Args:
            semantic_weight: Weight for semantic similarity component
            structural_weight: Weight for structural risk component
            dynamic_weight: Weight for dynamic usage component

        Note: Weights should sum to 1.0
        """
        self.w_sem = semantic_weight
        self.w_str = structural_weight
        self.w_dyn = dynamic_weight

        # Normalize weights
        total = self.w_sem + self.w_str + self.w_dyn
        if total > 0:
            self.w_sem /= total
            self.w_str /= total
            self.w_dyn /= total

    def compute_cycle_semantic_similarity(self,
                                         cycle_classes: List[str],
                                         embeddings: Dict[str, np.ndarray]) -> float:
        """
        Average pairwise cosine similarity within cycle.

        Args:
            cycle_classes: List of class names in the cycle
            embeddings: Dictionary mapping class names to embeddings

        Returns:
            Float in [0, 1], higher = more semantically coherent
        """
        return SimilarityCalculator.compute_cycle_similarity(cycle_classes, embeddings)

    def compute_structural_risk(self,
                               cycle_classes: List[str],
                               structural_metrics: pd.DataFrame,
                               cycle_edges: List[Tuple],
                               cbo_weight: float = 0.4,
                               instability_weight: float = 0.3,
                               size_weight: float = 0.3) -> float:
        """
        Structural risk score.

        Formula: 0.4 × norm(CBO) + 0.3 × norm(Instability) + 0.3 × norm(CycleSize)

        Args:
            cycle_classes: List of class names in the cycle
            structural_metrics: DataFrame with structural metrics for all classes
            cycle_edges: List of edges in the cycle
            cbo_weight: Weight for CBO component
            instability_weight: Weight for instability component
            size_weight: Weight for cycle size component

        Returns:
            Float in [0, 1], higher = more structurally risky
        """
        # Get metrics for cycle classes
        cycle_df = structural_metrics[structural_metrics['class_name'].isin(cycle_classes)]

        if cycle_df.empty:
            return 0.5  # Neutral score if no data

        # Compute components
        avg_cbo = cycle_df['CBO'].mean()
        avg_instability = cycle_df['Instability'].mean()
        cycle_size = len(cycle_edges)

        # Normalize (use project-wide max values)
        max_cbo = structural_metrics['CBO'].max()
        max_cycle_size = 50  # Reasonable upper bound for cycle size

        norm_cbo = min(avg_cbo / max_cbo, 1.0) if max_cbo > 0 else 0.0
        norm_instability = avg_instability  # Already in [0,1]
        norm_cycle_size = min(cycle_size / max_cycle_size, 1.0)

        # Compute weighted risk
        risk = (cbo_weight * norm_cbo +
               instability_weight * norm_instability +
               size_weight * norm_cycle_size)

        return float(risk)

    def compute_dynamic_score(self,
                            cycle_classes: List[str],
                            dynamic_data: Optional[Dict[str, float]] = None) -> float:
        """
        Dynamic usage frequency score.

        Args:
            cycle_classes: List of class names in the cycle
            dynamic_data: Dict mapping class_name -> usage_frequency [0, 1]
                         (e.g., based on commit frequency, test coverage)

        Returns:
            Average usage frequency (higher = more frequently used)
            Return 0.5 (neutral) if dynamic_data is None
        """
        if dynamic_data is None:
            return 0.5

        frequencies = [dynamic_data.get(cls, 0.5) for cls in cycle_classes]
        return float(np.mean(frequencies))

    def compute_anomaly_score(self,
                             cycle_classes: List[str],
                             cycle_edges: List[Tuple],
                             embeddings: Dict[str, np.ndarray],
                             structural_metrics: pd.DataFrame,
                             dynamic_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Compute final anomaly score.

        Anomaly formula:
        A = w_sem × (1 - semantic_similarity) +
            w_str × structural_risk +
            w_dyn × (1 - dynamic_score)

        Higher score = more anomalous = higher priority for refactoring

        Args:
            cycle_classes: List of class names in the cycle
            cycle_edges: List of (source, target, type) tuples
            embeddings: Dictionary of class embeddings
            structural_metrics: DataFrame with structural metrics
            dynamic_data: Optional dynamic usage data

        Returns:
            Dictionary with:
            - semantic_similarity: float
            - structural_risk: float
            - dynamic_score: float
            - anomaly_score: float
            - components: dict with individual components
        """
        # Compute components
        sem_sim = self.compute_cycle_semantic_similarity(cycle_classes, embeddings)
        str_risk = self.compute_structural_risk(cycle_classes, structural_metrics, cycle_edges)
        dyn_score = self.compute_dynamic_score(cycle_classes, dynamic_data)

        # Compute anomaly components
        # Low semantic similarity = anomalous (classes shouldn't be together)
        sem_component = self.w_sem * (1 - sem_sim)

        # High structural risk = anomalous
        str_component = self.w_str * str_risk

        # Low usage frequency = less critical to fix immediately
        dyn_component = self.w_dyn * (1 - dyn_score)

        # Total anomaly score
        anomaly = sem_component + str_component + dyn_component

        return {
            'semantic_similarity': sem_sim,
            'structural_risk': str_risk,
            'dynamic_score': dyn_score,
            'anomaly_score': anomaly,
            'components': {
                'semantic_component': sem_component,
                'structural_component': str_component,
                'dynamic_component': dyn_component
            }
        }

    def classify_priority(self, anomaly_score: float, percentiles: Dict[str, float]) -> str:
        """
        Classify cycle priority based on percentiles.

        Args:
            anomaly_score: Anomaly score for the cycle
            percentiles: Dict with 'P25', 'P50', 'P75', 'P90' keys

        Returns:
            Priority level: 'LOW', 'MEDIUM-LOW', 'MEDIUM-HIGH', 'HIGH', 'CRITICAL'
        """
        if anomaly_score < percentiles['P25']:
            return 'LOW'
        elif anomaly_score < percentiles['P50']:
            return 'MEDIUM-LOW'
        elif anomaly_score < percentiles['P75']:
            return 'MEDIUM-HIGH'
        elif anomaly_score < percentiles['P90']:
            return 'HIGH'
        else:
            return 'CRITICAL'

    def score_all_cycles(self,
                        cycles: List[Dict],
                        embeddings: Dict[str, np.ndarray],
                        structural_metrics: pd.DataFrame,
                        dynamic_data: Optional[Dict] = None) -> List[Dict]:
        """
        Compute anomaly scores for all cycles.

        Args:
            cycles: List of cycle dictionaries with 'cycle_classes' and 'cycle_edges'
            embeddings: Dictionary of class embeddings
            structural_metrics: DataFrame with structural metrics
            dynamic_data: Optional dynamic usage data

        Returns:
            List of cycles with added anomaly scores and components
        """
        scored_cycles = []

        for cycle in cycles:
            score_result = self.compute_anomaly_score(
                cycle['cycle_classes'],
                cycle['cycle_edges'],
                embeddings,
                structural_metrics,
                dynamic_data
            )

            # Merge with original cycle data
            scored_cycle = {**cycle, **score_result}
            scored_cycles.append(scored_cycle)

        # Compute percentiles
        anomaly_scores = [c['anomaly_score'] for c in scored_cycles]
        if anomaly_scores:
            percentiles = {
                'P25': float(np.percentile(anomaly_scores, 25)),
                'P50': float(np.percentile(anomaly_scores, 50)),
                'P75': float(np.percentile(anomaly_scores, 75)),
                'P90': float(np.percentile(anomaly_scores, 90))
            }

            # Classify priorities
            for cycle in scored_cycles:
                cycle['priority'] = self.classify_priority(
                    cycle['anomaly_score'],
                    percentiles
                )

        return scored_cycles

    def get_top_priority_cycles(self, scored_cycles: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Get the top-k highest priority cycles.

        Args:
            scored_cycles: List of cycles with anomaly scores
            top_k: Number of top cycles to return

        Returns:
            List of top-k cycles sorted by anomaly score (descending)
        """
        sorted_cycles = sorted(scored_cycles, key=lambda x: x['anomaly_score'], reverse=True)
        return sorted_cycles[:top_k]
