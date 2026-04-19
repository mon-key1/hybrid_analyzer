"""Structural metrics computation for Java classes and packages."""

from typing import Dict, List
import pandas as pd
import numpy as np


class StructuralMetrics:
    """Compute structural quality metrics for Java code."""

    @staticmethod
    def normalize_metric(values: pd.Series, max_value: float = None) -> pd.Series:
        """
        Normalize a metric to [0, 1] range.

        Args:
            values: Series of metric values
            max_value: Optional maximum value for normalization

        Returns:
            Normalized series
        """
        if max_value is None:
            max_value = values.max()

        if max_value == 0:
            return pd.Series([0.0] * len(values), index=values.index)

        return values / max_value

    @staticmethod
    def compute_percentiles(values: pd.Series) -> Dict[str, float]:
        """
        Compute percentiles for a series of values.

        Args:
            values: Series of metric values

        Returns:
            Dictionary with P25, P50, P75, P90 percentiles
        """
        return {
            'P25': float(values.quantile(0.25)),
            'P50': float(values.quantile(0.50)),
            'P75': float(values.quantile(0.75)),
            'P90': float(values.quantile(0.90))
        }

    @staticmethod
    def classify_by_percentile(value: float, percentiles: Dict[str, float]) -> str:
        """
        Classify a value based on percentiles.

        Args:
            value: Value to classify
            percentiles: Dictionary with P25, P50, P75, P90

        Returns:
            Classification: 'LOW', 'MEDIUM-LOW', 'MEDIUM-HIGH', 'HIGH', 'CRITICAL'
        """
        if value < percentiles['P25']:
            return 'LOW'
        elif value < percentiles['P50']:
            return 'MEDIUM-LOW'
        elif value < percentiles['P75']:
            return 'MEDIUM-HIGH'
        elif value < percentiles['P90']:
            return 'HIGH'
        else:
            return 'CRITICAL'

    @staticmethod
    def compute_package_metrics(structural_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute package-level metrics from class-level metrics.

        Args:
            structural_df: DataFrame with class-level metrics

        Returns:
            DataFrame with package-level metrics
        """
        package_metrics = []

        for package in structural_df['package'].unique():
            package_classes = structural_df[structural_df['package'] == package]

            metrics = {
                'package': package,
                'num_classes': len(package_classes),
                'avg_CBO': package_classes['CBO'].mean(),
                'max_CBO': package_classes['CBO'].max(),
                'avg_Instability': package_classes['Instability'].mean(),
                'max_Instability': package_classes['Instability'].max(),
                'total_dependencies': package_classes['Ce'].sum(),
                'total_dependents': package_classes['Ca'].sum()
            }

            package_metrics.append(metrics)

        return pd.DataFrame(package_metrics)

    @staticmethod
    def identify_god_classes(structural_df: pd.DataFrame,
                            cbo_threshold: float = 20,
                            methods_threshold: int = 50) -> pd.DataFrame:
        """
        Identify "God Classes" with high coupling and many methods.

        Args:
            structural_df: DataFrame with structural metrics
            cbo_threshold: CBO threshold for god class
            methods_threshold: Number of methods threshold

        Returns:
            DataFrame of potential god classes
        """
        god_classes = structural_df[
            (structural_df['CBO'] > cbo_threshold)
        ]

        return god_classes.sort_values('CBO', ascending=False)

    @staticmethod
    def compute_modularity_metrics(structural_df: pd.DataFrame) -> Dict:
        """
        Compute overall modularity metrics for the codebase.

        Args:
            structural_df: DataFrame with class-level metrics

        Returns:
            Dictionary with modularity metrics
        """
        return {
            'avg_CBO': float(structural_df['CBO'].mean()),
            'median_CBO': float(structural_df['CBO'].median()),
            'avg_Instability': float(structural_df['Instability'].mean()),
            'median_Instability': float(structural_df['Instability'].median()),
            'highly_coupled_classes': int((structural_df['CBO'] > 10).sum()),
            'highly_unstable_classes': int((structural_df['Instability'] > 0.7).sum()),
            'total_classes': len(structural_df)
        }
