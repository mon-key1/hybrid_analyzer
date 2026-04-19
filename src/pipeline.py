"""Main analysis pipeline orchestrating all components."""

import os
import glob
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

from .parser.java_parser import JavaParser, ParsedClass
from .parser.symbol_resolver import SymbolResolver
from .graph.dependency_graph import DependencyGraph
from .graph.cycle_detector import CycleDetector
from .semantic.embedder import CodeEmbedder
from .metrics.anomaly import AnomalyScorer
from .evaluation.baselines import BaselineMethods
from .evaluation.ranking_metrics import RankingMetrics
from .evaluation.statistical_tests import StatisticalTests


class CyclePrioritizationPipeline:
    """Main analysis pipeline for cycle prioritization."""

    def __init__(self, config: Dict):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Initialize components
        self.parser = JavaParser()
        self.symbol_resolver = SymbolResolver()

        # Initialize embedder
        self.embedder = CodeEmbedder(
            model_name=config.get('embedding_model', 'microsoft/unixcoder-base'),
            use_onnx=config.get('use_onnx', False)
        )

        # Initialize anomaly scorer
        self.anomaly_scorer = AnomalyScorer(
            semantic_weight=config.get('semantic_weight', 0.4),
            structural_weight=config.get('structural_weight', 0.4),
            dynamic_weight=config.get('dynamic_weight', 0.2)
        )

        self.graph = None
        self.cycle_detector = None

    def analyze_project(self, project_path: str) -> Dict:
        """
        Full analysis of a single project.

        Args:
            project_path: Path to Java project root

        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*80}")
        print(f"Analyzing project: {project_path}")
        print(f"{'='*80}\n")

        # Step 1: Parse Java files
        print("[1/7] Parsing Java files...")
        parsed_classes = self._parse_all_java_files(project_path)
        print(f"✓ Parsed {len(parsed_classes)} Java classes")

        if not parsed_classes:
            print("Warning: No Java classes found!")
            return self._empty_result(project_path)

        # Step 2: Build dependency graph
        print("\n[2/7] Building dependency graph...")
        self.graph = self._build_dependency_graph(parsed_classes)
        stats = self.graph.get_graph_stats()
        print(f"✓ Graph: {stats['num_classes']} classes, {stats['num_dependencies']} dependencies")

        # Step 3: Detect cycles
        print("\n[3/7] Detecting cycles...")
        self.cycle_detector = CycleDetector(self.graph)
        cycles = self.cycle_detector.find_all_cycles_with_classification()
        print(f"✓ Found {len(cycles)} cycles")

        if not cycles:
            print("No cycles detected!")
            return {
                'project_name': os.path.basename(project_path),
                'classes': parsed_classes,
                'dependency_graph': self.graph,
                'cycles': [],
                'structural_metrics': pd.DataFrame(),
                'embeddings': {},
                'anomaly_scores': [],
                'prioritized_cycles': []
            }

        # Step 4: Compute structural metrics
        print("\n[4/7] Computing structural metrics...")
        structural_metrics = self.graph.compute_structural_metrics()
        print(f"✓ Computed metrics for {len(structural_metrics)} classes")

        # Step 5: Generate semantic embeddings
        print("\n[5/7] Generating semantic embeddings...")
        embeddings = self._generate_embeddings(parsed_classes)
        print(f"✓ Generated {len(embeddings)} embeddings")

        # Step 6: Compute anomaly scores
        print("\n[6/7] Computing anomaly scores...")
        scored_cycles = self.anomaly_scorer.score_all_cycles(
            cycles,
            embeddings,
            structural_metrics,
            dynamic_data=None
        )
        print(f"✓ Scored {len(scored_cycles)} cycles")

        # Step 7: Rank cycles
        print("\n[7/7] Ranking cycles by anomaly score...")
        prioritized = sorted(scored_cycles, key=lambda x: x['anomaly_score'], reverse=True)
        print("✓ Cycles ranked")

        return {
            'project_name': os.path.basename(project_path),
            'classes': parsed_classes,
            'dependency_graph': self.graph,
            'cycles': cycles,
            'structural_metrics': structural_metrics,
            'embeddings': embeddings,
            'anomaly_scores': scored_cycles,
            'prioritized_cycles': [c['cycle_id'] for c in prioritized]
        }

    def _parse_all_java_files(self, project_path: str) -> List[ParsedClass]:
        """Parse all Java files in the project."""
        # Find all .java files
        pattern = os.path.join(project_path, '**', '*.java')
        java_files = glob.glob(pattern, recursive=True)

        print(f"Found {len(java_files)} Java files")

        parsed = []
        errors = 0

        for file_path in tqdm(java_files, desc="Parsing"):
            try:
                parsed_class = self.parser.parse_file(file_path)
                if parsed_class:
                    parsed.append(parsed_class)
            except Exception as e:
                errors += 1
                if errors <= 5:  # Show first 5 errors
                    print(f"Warning: Failed to parse {file_path}: {e}")

        if errors > 0:
            print(f"Warning: {errors} files failed to parse")

        return parsed

    def _build_dependency_graph(self, parsed_classes: List[ParsedClass]) -> DependencyGraph:
        """Build dependency graph from parsed classes."""
        graph = DependencyGraph()

        # Resolve dependencies
        dependency_map = self.symbol_resolver.resolve_all_dependencies(parsed_classes)

        # Add all classes as vertices
        for parsed_class in parsed_classes:
            metadata = {
                'parsed_class': parsed_class,
                'file_path': parsed_class.file_path
            }
            graph.add_class(
                parsed_class.fully_qualified_name,
                parsed_class.package_name,
                metadata
            )

        # Add edges
        for class_name, dependencies in dependency_map.items():
            for dep in dependencies:
                if dep in graph.class_to_vertex:  # Only add if target exists
                    graph.add_dependency(class_name, dep, 'uses')

        return graph

    def _generate_embeddings(self, parsed_classes: List[ParsedClass]) -> Dict[str, np.ndarray]:
        """Generate embeddings with caching."""
        cache_dir = self.config.get('cache_dir', 'data/embeddings_cache/')

        embeddings = self.embedder.embed_parsed_classes(
            parsed_classes,
            batch_size=16,
            cache_dir=cache_dir
        )

        return embeddings

    def _empty_result(self, project_path: str) -> Dict:
        """Return empty result structure."""
        return {
            'project_name': os.path.basename(project_path),
            'classes': [],
            'dependency_graph': DependencyGraph(),
            'cycles': [],
            'structural_metrics': pd.DataFrame(),
            'embeddings': {},
            'anomaly_scores': [],
            'prioritized_cycles': []
        }

    def evaluate_with_baselines(self,
                                analysis_result: Dict,
                                ground_truth_relevant: Optional[List[str]] = None) -> Dict:
        """
        Evaluate prioritization against baselines.

        Args:
            analysis_result: Result from analyze_project
            ground_truth_relevant: Optional list of relevant cycle IDs

        Returns:
            Dictionary with baseline comparison results
        """
        cycles = analysis_result['cycles']
        structural_metrics = analysis_result['structural_metrics']
        embeddings = analysis_result['embeddings']
        our_ranking = analysis_result['prioritized_cycles']

        # Compute all baseline rankings
        print("\nComputing baseline rankings...")
        baselines = BaselineMethods.get_all_baselines(
            cycles,
            structural_metrics,
            embeddings
        )

        # Add our method
        all_rankings = {
            'Semantic-Structural-Dynamic (Ours)': our_ranking,
            **baselines
        }

        if ground_truth_relevant is None:
            print("No ground truth provided - cannot compute ranking metrics")
            return {
                'baselines': baselines,
                'our_ranking': our_ranking
            }

        # Evaluate all rankings
        print("\nEvaluating rankings...")
        k = self.config.get('ndcg_k', 10)

        evaluation_results = {}
        for method_name, ranking in all_rankings.items():
            metrics = RankingMetrics.evaluate_ranking(
                ranking,
                ground_truth_relevant,
                k_values=[k]
            )
            evaluation_results[method_name] = metrics

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(evaluation_results).T

        return {
            'baselines': baselines,
            'our_ranking': our_ranking,
            'evaluation_results': evaluation_results,
            'comparison_df': comparison_df
        }

    def save_results(self, analysis_result: Dict, output_dir: str):
        """
        Save analysis results to files.

        Args:
            analysis_result: Result from analyze_project
            output_dir: Output directory
        """
        import json

        os.makedirs(output_dir, exist_ok=True)

        project_name = analysis_result['project_name']

        # Save anomaly scores as JSON
        scores_file = os.path.join(output_dir, f'{project_name}_anomaly_scores.json')
        with open(scores_file, 'w') as f:
            # Convert to serializable format
            serializable_scores = []
            for cycle in analysis_result['anomaly_scores']:
                cycle_copy = cycle.copy()
                # Remove non-serializable objects
                if 'cycle_edges' in cycle_copy:
                    cycle_copy['cycle_edges'] = [
                        {'source': e[0], 'target': e[1], 'type': e[2] if len(e) > 2 else 'uses'}
                        for e in cycle_copy['cycle_edges']
                    ]
                if 'classification' in cycle_copy and 'packages' in cycle_copy['classification']:
                    cycle_copy['classification']['packages'] = list(cycle_copy['classification']['packages'])

                serializable_scores.append(cycle_copy)

            json.dump(serializable_scores, f, indent=2)

        print(f"✓ Saved anomaly scores to {scores_file}")

        # Save structural metrics as CSV
        metrics_file = os.path.join(output_dir, f'{project_name}_structural_metrics.csv')
        analysis_result['structural_metrics'].to_csv(metrics_file, index=False)
        print(f"✓ Saved structural metrics to {metrics_file}")

        # Save prioritized cycles list
        ranking_file = os.path.join(output_dir, f'{project_name}_ranking.txt')
        with open(ranking_file, 'w') as f:
            f.write("Prioritized Cycles (High to Low Priority)\n")
            f.write("="*80 + "\n\n")

            for i, cycle in enumerate(analysis_result['anomaly_scores'][:20], 1):
                f.write(f"{i}. {cycle['cycle_id']} - {cycle['priority']}\n")
                f.write(f"   Anomaly Score: {cycle['anomaly_score']:.3f}\n")
                f.write(f"   Classes: {', '.join(cycle['cycle_classes'][:5])}")
                if len(cycle['cycle_classes']) > 5:
                    f.write(f" ... (+{len(cycle['cycle_classes'])-5} more)")
                f.write("\n\n")

        print(f"✓ Saved ranking to {ranking_file}")

    def print_summary(self, analysis_result: Dict):
        """Print analysis summary."""
        print(f"\n{'='*80}")
        print(f"ANALYSIS SUMMARY: {analysis_result['project_name']}")
        print(f"{'='*80}\n")

        print(f"Total Classes:      {len(analysis_result['classes'])}")
        print(f"Total Cycles:       {len(analysis_result['cycles'])}")

        if analysis_result['cycles']:
            scores = [c['anomaly_score'] for c in analysis_result['anomaly_scores']]
            print(f"\nAnomaly Score Statistics:")
            print(f"  Mean:    {np.mean(scores):.3f}")
            print(f"  Median:  {np.median(scores):.3f}")
            print(f"  Std:     {np.std(scores):.3f}")
            print(f"  Min:     {np.min(scores):.3f}")
            print(f"  Max:     {np.max(scores):.3f}")

            # Priority distribution
            priority_counts = pd.Series([c['priority'] for c in analysis_result['anomaly_scores']]).value_counts()
            print(f"\nPriority Distribution:")
            for priority in ['CRITICAL', 'HIGH', 'MEDIUM-HIGH', 'MEDIUM-LOW', 'LOW']:
                count = priority_counts.get(priority, 0)
                print(f"  {priority:12} {count:3d}")

            # Top 5 cycles
            print(f"\n{'='*80}")
            print("TOP 5 PRIORITY CYCLES")
            print(f"{'='*80}\n")

            for i, cycle in enumerate(analysis_result['anomaly_scores'][:5], 1):
                print(f"{i}. {cycle['cycle_id']} - {cycle['priority']}")
                print(f"   Anomaly Score: {cycle['anomaly_score']:.3f}")
                print(f"   Semantic Sim:  {cycle['semantic_similarity']:.3f}")
                print(f"   Struct Risk:   {cycle['structural_risk']:.3f}")
                print(f"   Type:          {cycle['classification']['type']}")
                print(f"   Classes:       {', '.join(cycle['cycle_classes'][:3])}")
                if len(cycle['cycle_classes']) > 3:
                    print(f"                  ... +{len(cycle['cycle_classes'])-3} more")
                print()
