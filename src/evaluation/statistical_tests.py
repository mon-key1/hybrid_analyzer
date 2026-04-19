"""Statistical significance tests for method comparison."""

from typing import Dict, Any
import numpy as np

try:
    from scipy import stats
except ImportError:
    print("Warning: scipy not installed. Install with: pip install scipy")
    stats = None


class StatisticalTests:
    """Statistical significance tests for comparing ranking methods."""

    @staticmethod
    def wilcoxon_signed_rank(scores_a: np.ndarray,
                            scores_b: np.ndarray) -> Dict[str, float]:
        """
        Wilcoxon signed-rank test for paired samples.

        Tests if two related paired samples come from the same distribution.
        Non-parametric alternative to paired t-test.

        Args:
            scores_a: First set of scores (e.g., proposed method)
            scores_b: Second set of scores (e.g., baseline method)

        Returns:
            Dictionary with:
            - statistic: Test statistic
            - p_value: Two-tailed p-value
        """
        if stats is None:
            raise ImportError("scipy required. Install with: pip install scipy")

        # Handle case where all differences are zero
        if np.all(scores_a == scores_b):
            return {'statistic': 0.0, 'p_value': 1.0}

        try:
            stat, p = stats.wilcoxon(scores_a, scores_b)
            return {'statistic': float(stat), 'p_value': float(p)}
        except ValueError as e:
            # Handle case with too few observations
            return {'statistic': 0.0, 'p_value': 1.0, 'error': str(e)}

    @staticmethod
    def cliffs_delta(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, Any]:
        """
        Cliff's delta effect size.

        Non-parametric effect size measure for comparing two groups.
        Ranges from -1 to +1:
        - +1: all values in A are larger than all values in B
        - -1: all values in A are smaller than all values in B
        - 0: overlapping distributions

        Args:
            scores_a: First set of scores
            scores_b: Second set of scores

        Returns:
            Dictionary with:
            - delta: Cliff's delta value
            - magnitude: Effect size magnitude
            - interpretation: Human-readable interpretation
        """
        n1, n2 = len(scores_a), len(scores_b)

        # Compute Cliff's delta
        dominance = sum(1 for a in scores_a for b in scores_b if a > b)
        ties = sum(1 for a in scores_a for b in scores_b if a == b)

        delta = (dominance - (n1 * n2 - dominance - ties)) / (n1 * n2)

        # Interpret magnitude (Romano et al., 2006)
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            magnitude = 'negligible'
        elif abs_delta < 0.33:
            magnitude = 'small'
        elif abs_delta < 0.474:
            magnitude = 'medium'
        else:
            magnitude = 'large'

        direction = 'A > B' if delta > 0 else 'B > A' if delta < 0 else 'A = B'

        return {
            'delta': float(delta),
            'magnitude': magnitude,
            'interpretation': f"{magnitude.capitalize()} effect size (|δ|={abs_delta:.3f}), {direction}"
        }

    @staticmethod
    def mann_whitney_u(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, float]:
        """
        Mann-Whitney U test (Wilcoxon rank-sum test).

        Tests if two independent samples come from the same distribution.

        Args:
            scores_a: First set of scores
            scores_b: Second set of scores

        Returns:
            Dictionary with statistic and p-value
        """
        if stats is None:
            raise ImportError("scipy required. Install with: pip install scipy")

        stat, p = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')

        return {'statistic': float(stat), 'p_value': float(p)}

    @staticmethod
    def paired_t_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, float]:
        """
        Paired t-test (parametric alternative to Wilcoxon).

        Args:
            scores_a: First set of scores
            scores_b: Second set of scores

        Returns:
            Dictionary with statistic and p-value
        """
        if stats is None:
            raise ImportError("scipy required. Install with: pip install scipy")

        stat, p = stats.ttest_rel(scores_a, scores_b)

        return {'statistic': float(stat), 'p_value': float(p)}

    @staticmethod
    def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, Any]:
        """
        Cohen's d effect size.

        Parametric effect size measure.

        Args:
            scores_a: First set of scores
            scores_b: Second set of scores

        Returns:
            Dictionary with Cohen's d and interpretation
        """
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)

        var_a = np.var(scores_a, ddof=1)
        var_b = np.var(scores_b, ddof=1)

        n_a = len(scores_a)
        n_b = len(scores_b)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

        # Cohen's d
        d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        # Interpret magnitude
        abs_d = abs(d)
        if abs_d < 0.2:
            magnitude = 'negligible'
        elif abs_d < 0.5:
            magnitude = 'small'
        elif abs_d < 0.8:
            magnitude = 'medium'
        else:
            magnitude = 'large'

        return {
            'd': float(d),
            'magnitude': magnitude,
            'interpretation': f"{magnitude.capitalize()} effect size (|d|={abs_d:.3f})"
        }

    @staticmethod
    def compare_methods(method_a_scores: np.ndarray,
                       method_b_scores: np.ndarray,
                       method_a_name: str = "Method A",
                       method_b_name: str = "Method B",
                       use_parametric: bool = False) -> Dict[str, Any]:
        """
        Comprehensive comparison of two methods.

        Args:
            method_a_scores: Scores for method A (across projects/folds)
            method_b_scores: Scores for method B (across projects/folds)
            method_a_name: Name of method A
            method_b_name: Name of method B
            use_parametric: Use parametric tests (t-test, Cohen's d)

        Returns:
            Dictionary with test results and interpretation
        """
        results = {
            'method_a': method_a_name,
            'method_b': method_b_name,
            'mean_a': float(np.mean(method_a_scores)),
            'mean_b': float(np.mean(method_b_scores)),
            'std_a': float(np.std(method_a_scores)),
            'std_b': float(np.std(method_b_scores)),
            'improvement': float((np.mean(method_a_scores) - np.mean(method_b_scores)) /
                                np.mean(method_b_scores) * 100) if np.mean(method_b_scores) != 0 else 0.0
        }

        # Statistical significance test
        if use_parametric:
            results['significance_test'] = StatisticalTests.paired_t_test(
                method_a_scores, method_b_scores
            )
            results['effect_size'] = StatisticalTests.cohens_d(
                method_a_scores, method_b_scores
            )
        else:
            results['significance_test'] = StatisticalTests.wilcoxon_signed_rank(
                method_a_scores, method_b_scores
            )
            results['effect_size'] = StatisticalTests.cliffs_delta(
                method_a_scores, method_b_scores
            )

        # Interpretation
        p_value = results['significance_test']['p_value']
        is_significant = p_value < 0.05

        results['is_significant'] = is_significant
        results['interpretation'] = (
            f"{method_a_name} is {'significantly' if is_significant else 'not significantly'} "
            f"different from {method_b_name} (p={p_value:.4f}). "
            f"Effect size: {results['effect_size']['interpretation']}. "
            f"Improvement: {results['improvement']:.1f}%."
        )

        return results
