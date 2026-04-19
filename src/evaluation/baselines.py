"""Baseline prioritization methods for comparison."""

import random
from typing import List, Dict
import pandas as pd
import numpy as np

from ..semantic.similarity import SimilarityCalculator


class BaselineMethods:
    """Baseline prioritization methods for cycle ranking."""

    @staticmethod
    def random_baseline(cycles: List[Dict], seed: int = 42) -> List[str]:
        """
        Random ordering baseline.

        Args:
            cycles: List of cycle dictionaries
            seed: Random seed for reproducibility

        Returns:
            List of cycle IDs in random order
        """
        random.seed(seed)
        cycle_ids = [c['cycle_id'] for c in cycles]
        shuffled = cycle_ids.copy()
        random.shuffle(shuffled)
        return shuffled

    @staticmethod
    def cbo_baseline(cycles: List[Dict], structural_metrics: pd.DataFrame) -> List[str]:
        """
        Prioritize by average CBO (Coupling Between Objects) - descending.

        Higher CBO = higher coupling = higher priority.

        Args:
            cycles: List of cycle dictionaries
            structural_metrics: DataFrame with structural metrics

        Returns:
            List of cycle IDs sorted by average CBO (descending)
        """
        scored = []

        for cycle in cycles:
            classes = cycle['cycle_classes']
            cycle_df = structural_metrics[structural_metrics['class_name'].isin(classes)]

            if not cycle_df.empty:
                avg_cbo = cycle_df['CBO'].mean()
            else:
                avg_cbo = 0.0

            scored.append((cycle['cycle_id'], avg_cbo))

        # Sort by CBO descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [cycle_id for cycle_id, _ in scored]

    @staticmethod
    def instability_baseline(cycles: List[Dict], structural_metrics: pd.DataFrame) -> List[str]:
        """
        Prioritize by average Instability - descending.

        Higher instability = more likely to change = higher priority.

        Args:
            cycles: List of cycle dictionaries
            structural_metrics: DataFrame with structural metrics

        Returns:
            List of cycle IDs sorted by average instability (descending)
        """
        scored = []

        for cycle in cycles:
            classes = cycle['cycle_classes']
            cycle_df = structural_metrics[structural_metrics['class_name'].isin(classes)]

            if not cycle_df.empty:
                avg_inst = cycle_df['Instability'].mean()
            else:
                avg_inst = 0.0

            scored.append((cycle['cycle_id'], avg_inst))

        # Sort by instability descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [cycle_id for cycle_id, _ in scored]

    @staticmethod
    def cycle_size_baseline(cycles: List[Dict]) -> List[str]:
        """
        Prioritize by cycle size (number of edges) - descending.

        Larger cycles = more complex = higher priority.

        Args:
            cycles: List of cycle dictionaries

        Returns:
            List of cycle IDs sorted by size (descending)
        """
        scored = [(c['cycle_id'], len(c['cycle_edges'])) for c in cycles]

        # Sort by size descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [cycle_id for cycle_id, _ in scored]

    @staticmethod
    def semantic_only_baseline(cycles: List[Dict],
                               embeddings: Dict[str, np.ndarray]) -> List[str]:
        """
        Prioritize by semantic similarity - ascending.

        Low semantic similarity = classes shouldn't be together = higher priority.

        Args:
            cycles: List of cycle dictionaries
            embeddings: Dictionary of class embeddings

        Returns:
            List of cycle IDs sorted by semantic similarity (ascending)
        """
        scored = []

        for cycle in cycles:
            sem_sim = SimilarityCalculator.compute_cycle_similarity(
                cycle['cycle_classes'],
                embeddings
            )
            scored.append((cycle['cycle_id'], sem_sim))

        # Sort by similarity ascending (low similarity = high priority)
        scored.sort(key=lambda x: x[1])

        return [cycle_id for cycle_id, _ in scored]

    @staticmethod
    def structural_only_baseline(cycles: List[Dict],
                                structural_metrics: pd.DataFrame) -> List[str]:
        """
        Prioritize by structural risk only (CBO + Instability + Size).

        Equivalent to anomaly scorer with semantic_weight=0.

        Args:
            cycles: List of cycle dictionaries
            structural_metrics: DataFrame with structural metrics

        Returns:
            List of cycle IDs sorted by structural risk (descending)
        """
        from ..metrics.anomaly import AnomalyScorer

        # Create scorer with only structural weight
        scorer = AnomalyScorer(
            semantic_weight=0.0,
            structural_weight=0.7,
            dynamic_weight=0.3
        )

        scored = []

        for cycle in cycles:
            # Compute structural risk
            str_risk = scorer.compute_structural_risk(
                cycle['cycle_classes'],
                structural_metrics,
                cycle['cycle_edges']
            )

            scored.append((cycle['cycle_id'], str_risk))

        # Sort by risk descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [cycle_id for cycle_id, _ in scored]

    @staticmethod
    def cross_package_first_baseline(cycles: List[Dict]) -> List[str]:
        """
        Prioritize cross-package cycles first, then intra-package.

        Within each category, sort by size.

        Args:
            cycles: List of cycle dictionaries

        Returns:
            List of cycle IDs
        """
        cross_package = []
        intra_package = []

        for cycle in cycles:
            cycle_type = cycle.get('classification', {}).get('type', 'unknown')
            size = len(cycle['cycle_edges'])

            if cycle_type == 'cross-package':
                cross_package.append((cycle['cycle_id'], size))
            else:
                intra_package.append((cycle['cycle_id'], size))

        # Sort each category by size descending
        cross_package.sort(key=lambda x: x[1], reverse=True)
        intra_package.sort(key=lambda x: x[1], reverse=True)

        # Combine: cross-package first
        result = ([cid for cid, _ in cross_package] +
                 [cid for cid, _ in intra_package])

        return result

    @staticmethod
    def combined_structural_baseline(cycles: List[Dict],
                                    structural_metrics: pd.DataFrame) -> List[str]:
        """
        Combined structural baseline: (CBO + Instability) / 2.

        Args:
            cycles: List of cycle dictionaries
            structural_metrics: DataFrame with structural metrics

        Returns:
            List of cycle IDs sorted by combined metric (descending)
        """
        scored = []

        for cycle in cycles:
            classes = cycle['cycle_classes']
            cycle_df = structural_metrics[structural_metrics['class_name'].isin(classes)]

            if not cycle_df.empty:
                avg_cbo = cycle_df['CBO'].mean()
                avg_inst = cycle_df['Instability'].mean()

                # Normalize CBO to [0, 1]
                max_cbo = structural_metrics['CBO'].max()
                norm_cbo = avg_cbo / max_cbo if max_cbo > 0 else 0.0

                # Combined score
                combined = (norm_cbo + avg_inst) / 2.0
            else:
                combined = 0.0

            scored.append((cycle['cycle_id'], combined))

        # Sort descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [cycle_id for cycle_id, _ in scored]

    @staticmethod
    def get_all_baselines(cycles: List[Dict],
                         structural_metrics: pd.DataFrame,
                         embeddings: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """
        Compute all baseline rankings.

        Args:
            cycles: List of cycle dictionaries
            structural_metrics: DataFrame with structural metrics
            embeddings: Dictionary of class embeddings

        Returns:
            Dictionary mapping baseline name to ranked cycle IDs
        """
        return {
            'Random': BaselineMethods.random_baseline(cycles),
            'CBO': BaselineMethods.cbo_baseline(cycles, structural_metrics),
            'Instability': BaselineMethods.instability_baseline(cycles, structural_metrics),
            'Cycle Size': BaselineMethods.cycle_size_baseline(cycles),
            'Semantic Only': BaselineMethods.semantic_only_baseline(cycles, embeddings),
            'Structural Only': BaselineMethods.structural_only_baseline(cycles, structural_metrics),
            'Cross-Package First': BaselineMethods.cross_package_first_baseline(cycles),
            'Combined Structural': BaselineMethods.combined_structural_baseline(cycles, structural_metrics)
        }
