"""Automatic cycle validation against git history."""

from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
import re
import os

try:
    from pydriller import Repository
except ImportError:
    Repository = None

from ..mining.git_analyzer import GitHistoryAnalyzer
from .ranking_metrics import RankingMetrics


class CycleValidator:
    """Validate cycle prioritization by analyzing git history."""

    def __init__(self, project_path: str, git_path: Optional[str] = None):
        """
        Initialize cycle validator.

        Args:
            project_path: Path to Java project
            git_path: Path to git repository (defaults to project_path)
        """
        self.project_path = project_path
        self.git_path = git_path or project_path

        if Repository is None:
            raise ImportError("pydriller required. Install with: pip install pydriller")

        self.git_analyzer = GitHistoryAnalyzer(self.git_path)
        self.repo = Repository(self.git_path)

    def find_eliminated_cycles(self,
                               all_cycles: List[Dict],
                               since: Optional[str] = None,
                               until: Optional[str] = None) -> List[Dict]:
        """
        Find cycles that were eliminated in git history.

        A cycle is considered "eliminated" if:
        1. A commit removes/refactors dependencies between classes in the cycle
        2. The commit message mentions refactoring/cycle/circular keywords
        3. The changes reduce coupling between cycle participants

        Args:
            all_cycles: List of all detected cycles (from cycle detector)
            since: Start date for git history search (ISO format)
            until: End date for git history search

        Returns:
            List of eliminated cycles with metadata:
            - cycle_id: str
            - cycle_classes: List[str]
            - elimination_commit: str (commit hash)
            - elimination_date: datetime
            - commit_message: str
        """
        eliminated_cycles = []

        # Get refactoring commits
        refactoring_commits = self.git_analyzer.find_refactoring_commits(
            since=since, until=until
        )

        # For each cycle, check if it was eliminated
        for cycle in all_cycles:
            cycle_classes = cycle['cycle_classes']

            # Check each refactoring commit
            for commit_data in refactoring_commits:
                if self._is_cycle_eliminated_in_commit(
                    cycle_classes,
                    commit_data
                ):
                    eliminated_cycles.append({
                        'cycle_id': cycle['cycle_id'],
                        'cycle_classes': cycle_classes,
                        'elimination_commit': commit_data['hash'],
                        'elimination_date': commit_data['date'],
                        'commit_message': commit_data['message'],
                        'keywords': commit_data.get('keywords', [])
                    })
                    break  # Only count first elimination

        return eliminated_cycles

    def _is_cycle_eliminated_in_commit(self,
                                       cycle_classes: List[str],
                                       commit_data: Dict) -> bool:
        """
        Check if a cycle was eliminated in a specific commit.

        Heuristics:
        1. At least 2 classes from the cycle were modified
        2. Commit message mentions cycle/refactoring keywords
        3. Changes involve dependency removal (imports, method calls)

        Args:
            cycle_classes: List of class names in the cycle
            commit_data: Commit metadata from git_analyzer

        Returns:
            True if cycle was likely eliminated
        """
        # Extract simple class names (without package)
        simple_cycle_classes = [self._extract_class_name(cls) for cls in cycle_classes]

        # Check if files related to cycle classes were modified
        modified_files = commit_data.get('files_changed', [])

        # Extract class names from file paths
        modified_classes = [
            self._extract_class_name_from_file(f)
            for f in modified_files
        ]

        # Count how many cycle classes were modified
        modified_cycle_classes = set(simple_cycle_classes) & set(modified_classes)

        # Heuristic 1: At least 2 classes from cycle were modified
        if len(modified_cycle_classes) < 2:
            return False

        # Heuristic 2: Commit mentions cycle or refactoring
        mentions_cycle = commit_data.get('mentions_cycle', False)
        is_refactoring = commit_data.get('is_refactoring', False)

        if mentions_cycle or is_refactoring:
            return True

        # Heuristic 3: Check if deletions > insertions (suggesting removal)
        deletions = commit_data.get('deletions', 0)
        insertions = commit_data.get('insertions', 0)

        if deletions > insertions and deletions > 10:
            return True

        return False

    def _extract_class_name(self, full_class_name: str) -> str:
        """
        Extract simple class name from fully qualified name.

        Args:
            full_class_name: e.g., "com.example.MyClass"

        Returns:
            Simple name: e.g., "MyClass"
        """
        return full_class_name.split('.')[-1]

    def _extract_class_name_from_file(self, file_path: str) -> str:
        """
        Extract class name from file path.

        Args:
            file_path: e.g., "src/main/java/com/example/MyClass.java"

        Returns:
            Class name: e.g., "MyClass"
        """
        if not file_path or not file_path.endswith('.java'):
            return ''

        basename = os.path.basename(file_path)
        return basename.replace('.java', '')

    def validate_ranking(self,
                        ranked_cycles: List[Dict],
                        eliminated_cycles: List[Dict],
                        k: int = 10) -> Dict[str, float]:
        """
        Validate cycle ranking against eliminated cycles.

        Args:
            ranked_cycles: Cycles ranked by anomaly score (our prediction)
            eliminated_cycles: Cycles actually eliminated in git history (ground truth)
            k: Top-k to evaluate

        Returns:
            Dictionary with validation metrics:
            - NDCG@k: Ranking quality
            - Precision@k: How many top-k were actually eliminated
            - Recall@k: How many eliminated cycles found in top-k
            - F1@k: Harmonic mean
            - total_eliminated: Number of eliminated cycles
            - found_in_top_k: Number of eliminated cycles in top-k
        """
        # Extract ranked cycle IDs
        ranked_ids = [cycle['cycle_id'] for cycle in ranked_cycles]

        # Extract eliminated cycle IDs (ground truth)
        eliminated_ids = [cycle['cycle_id'] for cycle in eliminated_cycles]
        eliminated_set = set(eliminated_ids)

        # Compute metrics
        metrics = RankingMetrics.evaluate_ranking(
            ranked_items=ranked_ids,
            relevant_items=eliminated_ids,
            k_values=[k]
        )

        # Add custom metrics
        top_k_ids = ranked_ids[:k]
        found_in_top_k = len([cid for cid in top_k_ids if cid in eliminated_set])

        metrics.update({
            'total_eliminated': len(eliminated_ids),
            'found_in_top_k': found_in_top_k,
            'total_cycles': len(ranked_cycles),
            'k': k
        })

        return metrics

    def generate_validation_report(self,
                                   ranked_cycles: List[Dict],
                                   eliminated_cycles: List[Dict],
                                   metrics: Dict[str, float],
                                   k: int = 10) -> str:
        """
        Generate human-readable validation report.

        Args:
            ranked_cycles: Ranked cycles
            eliminated_cycles: Eliminated cycles from git history
            metrics: Validation metrics
            k: Top-k

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CYCLE VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary
        report_lines.append("Summary:")
        report_lines.append(f"  Total cycles detected:        {metrics['total_cycles']}")
        report_lines.append(f"  Total cycles eliminated:      {metrics['total_eliminated']}")
        report_lines.append(f"  Eliminated cycles in top-{k}:   {metrics['found_in_top_k']}")
        report_lines.append("")

        # Metrics
        report_lines.append("Validation Metrics:")
        report_lines.append(f"  NDCG@{k}:       {metrics.get(f'NDCG@{k}', 0):.4f}")
        report_lines.append(f"  Precision@{k}:  {metrics.get(f'Precision@{k}', 0):.4f}")
        report_lines.append(f"  Recall@{k}:     {metrics.get(f'Recall@{k}', 0):.4f}")
        report_lines.append(f"  F1@{k}:         {metrics.get(f'F1@{k}', 0):.4f}")
        report_lines.append("")

        # Interpretation
        report_lines.append("Interpretation:")
        ndcg = metrics.get(f'NDCG@{k}', 0)
        precision = metrics.get(f'Precision@{k}', 0)

        if ndcg >= 0.75:
            report_lines.append("  ✓ Excellent ranking quality (NDCG ≥ 0.75)")
        elif ndcg >= 0.60:
            report_lines.append("  ~ Good ranking quality (NDCG ≥ 0.60)")
        else:
            report_lines.append("  ✗ Poor ranking quality (NDCG < 0.60)")

        if precision >= 0.60:
            report_lines.append("  ✓ High precision (≥ 60% of top-k are problematic)")
        elif precision >= 0.40:
            report_lines.append("  ~ Moderate precision (≥ 40% of top-k are problematic)")
        else:
            report_lines.append("  ✗ Low precision (< 40% of top-k are problematic)")

        report_lines.append("")

        # Top-k analysis
        report_lines.append(f"Top-{k} Cycles Analysis:")
        report_lines.append("")

        eliminated_set = {cycle['cycle_id'] for cycle in eliminated_cycles}

        for i, cycle in enumerate(ranked_cycles[:k], 1):
            cycle_id = cycle['cycle_id']
            is_eliminated = cycle_id in eliminated_set

            status = "✓ ELIMINATED" if is_eliminated else "  (still exists)"
            anomaly_score = cycle.get('anomaly_score', 0)

            report_lines.append(
                f"  {i:2d}. {cycle_id} - Score: {anomaly_score:.3f} {status}"
            )

            if is_eliminated:
                # Find elimination details
                elim_info = next(
                    (e for e in eliminated_cycles if e['cycle_id'] == cycle_id),
                    None
                )
                if elim_info:
                    commit_hash = elim_info['elimination_commit'][:8]
                    date = elim_info['elimination_date'].strftime('%Y-%m-%d')
                    report_lines.append(
                        f"       → Eliminated in {commit_hash} on {date}"
                    )

        report_lines.append("")

        # Eliminated cycles not in top-k
        not_found = [
            cycle for cycle in eliminated_cycles
            if cycle['cycle_id'] not in [c['cycle_id'] for c in ranked_cycles[:k]]
        ]

        if not_found:
            report_lines.append(f"Eliminated Cycles NOT in Top-{k}:")
            for cycle in not_found[:5]:  # Show first 5
                cycle_id = cycle['cycle_id']
                commit_hash = cycle['elimination_commit'][:8]
                date = cycle['elimination_date'].strftime('%Y-%m-%d')
                report_lines.append(
                    f"  - {cycle_id} (eliminated in {commit_hash} on {date})"
                )

            if len(not_found) > 5:
                report_lines.append(f"  ... and {len(not_found) - 5} more")

        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def run_validation(self,
                       analysis_results: Dict,
                       since: Optional[str] = None,
                       until: Optional[str] = None,
                       k: int = 10) -> Dict:
        """
        Run complete validation pipeline.

        Args:
            analysis_results: Results from pipeline.analyze_project()
            since: Start date for git history
            until: End date for git history
            k: Top-k to evaluate

        Returns:
            Dictionary with:
            - metrics: Validation metrics
            - eliminated_cycles: List of eliminated cycles
            - report: Human-readable report
        """
        # Get all cycles
        all_cycles = analysis_results.get('cycles', [])

        # Get ranked cycles (by anomaly score)
        ranked_cycles = sorted(
            analysis_results.get('anomaly_scores', []),
            key=lambda x: x.get('anomaly_score', 0),
            reverse=True
        )

        # Find eliminated cycles in git history
        eliminated_cycles = self.find_eliminated_cycles(
            all_cycles,
            since=since,
            until=until
        )

        # Compute validation metrics
        metrics = self.validate_ranking(
            ranked_cycles,
            eliminated_cycles,
            k=k
        )

        # Generate report
        report = self.generate_validation_report(
            ranked_cycles,
            eliminated_cycles,
            metrics,
            k=k
        )

        return {
            'metrics': metrics,
            'eliminated_cycles': eliminated_cycles,
            'ranked_cycles': ranked_cycles,
            'report': report
        }
