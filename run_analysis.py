#!/usr/bin/env python3
"""
Semantic Cycle Prioritization System - CLI Entry Point

Usage:
    python run_analysis.py analyze --project <path>
    python run_analysis.py validate --project <path> --git <git_path>
    python run_analysis.py compare --project <path>
"""

import os
import sys
import yaml
import click
import tempfile
import shutil
import subprocess
from datetime import datetime

from src.pipeline import CyclePrioritizationPipeline
from src.evaluation.baselines import BaselineMethods
from src.evaluation.cycle_validator import CycleValidator


def convert_to_serializable(obj):
    """Recursively convert non-serializable objects (sets, numpy types) to JSON-compatible types."""
    import numpy as np

    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # Fallback for other types
        return str(obj)


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    Semantic Cycle Prioritization System

    Detect, prioritize, and validate cyclic dependencies in Java projects
    using semantic, structural, and dynamic analysis.
    """
    pass


def _resolve_commit_ref(git_path: str, commit_ref: str) -> str:
    """Resolve commit reference (hash or date) to commit hash."""
    try:
        target_date = datetime.strptime(commit_ref, '%Y-%m-%d')
        # Find commit before this date
        result = subprocess.run(
            ['git', 'log', '--until', commit_ref, '--format=%H %ci', '-1'],
            cwd=git_path, capture_output=True, text=True, check=True
        )
        commit_before = result.stdout.strip()
        # Find commit after this date
        result = subprocess.run(
            ['git', 'log', '--since', commit_ref, '--reverse', '--format=%H %ci', '-1'],
            cwd=git_path, capture_output=True, text=True, check=True
        )
        commit_after = result.stdout.strip()
        # Choose closest
        if commit_before and commit_after:
            before_hash, before_date_str = commit_before.split(' ', 1)
            after_hash, after_date_str = commit_after.split(' ', 1)
            before_date = datetime.fromisoformat(before_date_str.split(' ')[0])
            after_date = datetime.fromisoformat(after_date_str.split(' ')[0])
            before_diff = abs((target_date - before_date).total_seconds())
            after_diff = abs((after_date - target_date).total_seconds())
            return before_hash if before_diff <= after_diff else after_hash
        elif commit_before:
            return commit_before.split(' ')[0]
        elif commit_after:
            return commit_after.split(' ')[0]
        else:
            raise ValueError(f"No commits found near {commit_ref}")
    except ValueError as e:
        if 'does not match format' in str(e) or 'unconverted data remains' in str(e):
            result = subprocess.run(
                ['git', 'rev-parse', '--verify', commit_ref],
                cwd=git_path, capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        else:
            raise


def _checkout_at_commit(git_path: str, project_subdir: str, commit_hash: str) -> str:
    """Create temporary copy of project at specific commit."""
    temp_dir = tempfile.mkdtemp(prefix='cycle_analysis_')
    try:
        worktree_path = os.path.join(temp_dir, 'worktree')
        subprocess.run(
            ['git', 'worktree', 'add', '--detach', worktree_path, commit_hash],
            cwd=git_path, capture_output=True, text=True, check=True
        )
        if project_subdir:
            return os.path.join(worktree_path, project_subdir)
        else:
            return worktree_path
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to checkout commit: {e.stderr}")


def _cleanup_worktree(git_path: str, worktree_path: str):
    """Clean up git worktree."""
    try:
        subprocess.run(
            ['git', 'worktree', 'remove', '--force', worktree_path],
            cwd=git_path, capture_output=True, text=True
        )
    except:
        pass
    temp_parent = os.path.dirname(worktree_path)
    if os.path.exists(temp_parent) and 'cycle_analysis_' in temp_parent:
        shutil.rmtree(temp_parent, ignore_errors=True)


@cli.command()
@click.option('--project', required=True, type=click.Path(exists=True),
              help='Path to Java project directory')
@click.option('--config', default='config.yaml', type=click.Path(exists=True),
              help='Configuration file (default: config.yaml)')
@click.option('--output', default=None, type=click.Path(),
              help='Output directory (default: from config)')
@click.option('--at-commit', default=None, type=str,
              help='Analyze project at specific commit (hash or date YYYY-MM-DD)')
def analyze(project, config, output, at_commit):
    """
    Analyze a single Java project for cyclic dependencies.

    This command will:
    1. Parse all Java files in the project
    2. Build a dependency graph
    3. Detect cycles using Tarjan's algorithm
    4. Compute semantic embeddings
    5. Calculate anomaly scores
    6. Rank cycles by priority

    Example:
        python run_analysis.py analyze --project /path/to/java/project
        python run_analysis.py analyze --project /path/to/java/project --at-commit 2015-01-01
    """
    # Load configuration
    with open(config) as f:
        cfg = yaml.safe_load(f)

    # Override output directory if specified
    if output:
        cfg['output_dir'] = output

    # Handle --at-commit option
    git_root = None
    worktree_path = None
    project_to_analyze = project

    if at_commit:
        # Auto-detect git root
        current_path = os.path.abspath(project)
        for _ in range(5):
            if os.path.exists(os.path.join(current_path, '.git')):
                git_root = current_path
                break
            parent = os.path.dirname(current_path)
            if parent == current_path:
                break
            current_path = parent

        if not git_root:
            click.echo("Error: --at-commit requires a git repository.", err=True)
            sys.exit(1)

        click.echo(f"Checking out project at commit/date: {at_commit}...")
        commit_hash = _resolve_commit_ref(git_root, at_commit)

        # Get commit date
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ci', commit_hash],
            cwd=git_root, capture_output=True, text=True, check=True
        )
        commit_date = result.stdout.strip().split(' ')[0]
        click.echo(f"  Resolved to commit: {commit_hash[:8]} (date: {commit_date})")

        # Determine project subdirectory
        git_root_abs = os.path.abspath(git_root)
        project_abs = os.path.abspath(project)
        if project_abs.startswith(git_root_abs):
            project_subdir = os.path.relpath(project_abs, git_root_abs)
            if project_subdir == '.':
                project_subdir = ''
        else:
            project_subdir = ''

        # Checkout
        worktree_path = _checkout_at_commit(git_root, project_subdir, commit_hash)
        project_to_analyze = worktree_path
        click.echo(f"  Analyzing project at: {project_to_analyze}\n")

    # Create pipeline
    pipeline = CyclePrioritizationPipeline(cfg)

    # Run analysis
    try:
        results = pipeline.analyze_project(project_to_analyze)

        # Print summary
        pipeline.print_summary(results)

        # Save results
        output_dir = cfg.get('output_dir', 'data/results/')
        pipeline.save_results(results, output_dir)

        click.echo(f"\n✓ Analysis complete!")
        click.echo(f"Results saved to: {output_dir}")

    except Exception as e:
        click.echo(f"Error during analysis: {e}", err=True)
        if cfg.get('verbose', True):
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up worktree
        if worktree_path:
            click.echo("\nCleaning up temporary worktree...")
            _cleanup_worktree(git_root, worktree_path)


@cli.command()
@click.option('--project', required=True, type=click.Path(exists=True),
              help='Path to Java project directory')
@click.option('--config', default='config.yaml', type=click.Path(exists=True),
              help='Configuration file')
@click.option('--output', default=None, type=click.Path(),
              help='Output directory')
def compare(project, config, output):
    """
    Analyze a project and compare with baseline methods.

    This command runs the full analysis and compares the semantic-structural-dynamic
    prioritization against multiple baseline methods (CBO, Instability, Size, etc.).

    Example:
        python run_analysis.py compare --project /path/to/java/project
    """
    # Load configuration
    with open(config) as f:
        cfg = yaml.safe_load(f)

    if output:
        cfg['output_dir'] = output

    # Create pipeline
    pipeline = CyclePrioritizationPipeline(cfg)

    try:
        # Run analysis
        results = pipeline.analyze_project(project)

        # Evaluate with baselines
        click.echo("\n" + "="*80)
        click.echo("BASELINE COMPARISON")
        click.echo("="*80 + "\n")

        baseline_results = pipeline.evaluate_with_baselines(results)

        if 'comparison_df' in baseline_results:
            click.echo(baseline_results['comparison_df'].to_string())
        else:
            click.echo("Baseline comparison computed (no ground truth for metrics)")

        # Save results
        output_dir = cfg.get('output_dir', 'data/results/')
        pipeline.save_results(results, output_dir)

        # Save baseline comparison
        import json
        baseline_file = os.path.join(output_dir,
                                    f"{results['project_name']}_baselines.json")
        with open(baseline_file, 'w') as f:
            # Convert to serializable format
            serializable = {
                'our_ranking': baseline_results['our_ranking'],
                'baselines': baseline_results['baselines']
            }
            json.dump(serializable, f, indent=2)

        click.echo(f"\n✓ Comparison complete!")
        click.echo(f"Results saved to: {output_dir}")

    except Exception as e:
        click.echo(f"Error during comparison: {e}", err=True)
        if cfg.get('verbose', True):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--project', required=True, type=click.Path(exists=True),
              help='Path to Java project directory')
@click.option('--git', default=None, type=click.Path(exists=True),
              help='Path to git repository (defaults to project path)')
@click.option('--config', default='config.yaml', type=click.Path(exists=True),
              help='Configuration file')
@click.option('--output', default=None, type=click.Path(),
              help='Output directory')
@click.option('--since', default=None, type=str,
              help='Start date for git history (YYYY-MM-DD)')
@click.option('--until', default=None, type=str,
              help='End date for git history (YYYY-MM-DD)')
@click.option('--top-k', default=10, type=int,
              help='Number of top cycles to evaluate (default: 10)')
def validate(project, git, config, output, since, until, top_k):
    """
    Validate cycle prioritization against git history.

    This command:
    1. Analyzes the project to detect and rank cycles
    2. Searches git history for commits that eliminated cycles
    3. Compares top-k ranked cycles with actually eliminated cycles
    4. Computes validation metrics (NDCG@k, Precision@k, Recall@k)

    The validation answers: "Did our system correctly identify the most
    problematic cycles that were later fixed?"

    Example:
        python run_analysis.py validate --project /path/to/java/project \\
            --since 2020-01-01 --top-k 10
    """
    # Load configuration
    with open(config) as f:
        cfg = yaml.safe_load(f)

    if output:
        cfg['output_dir'] = output

    try:
        # Step 1: Analyze project
        click.echo("Step 1: Analyzing project for cyclic dependencies...")
        pipeline = CyclePrioritizationPipeline(cfg)
        results = pipeline.analyze_project(project)

        click.echo(f"  Found {len(results.get('cycles', []))} cycles")
        click.echo(f"  Ranked by anomaly score\n")

        # Step 2: Create validator and run validation
        click.echo("Step 2: Searching git history for eliminated cycles...")
        validator = CycleValidator(project, git_path=git)

        validation_results = validator.run_validation(
            results,
            since=since,
            until=until,
            k=top_k
        )

        eliminated_count = len(validation_results['eliminated_cycles'])
        click.echo(f"  Found {eliminated_count} eliminated cycles in git history\n")

        # Step 3: Display validation report
        click.echo(validation_results['report'])

        # Step 4: Save results
        output_dir = cfg.get('output_dir', 'data/results/')
        os.makedirs(output_dir, exist_ok=True)

        # Save full validation results
        import json
        validation_file = os.path.join(
            output_dir,
            f"{results['project_name']}_validation.json"
        )

        with open(validation_file, 'w') as f:
            # Prepare serializable data
            serializable = {
                'metrics': validation_results['metrics'],
                'eliminated_cycles': [
                    {
                        'cycle_id': c['cycle_id'],
                        'cycle_classes': c['cycle_classes'],
                        'elimination_commit': c['elimination_commit'],
                        'elimination_date': c['elimination_date'].isoformat(),
                        'commit_message': c['commit_message']
                    }
                    for c in validation_results['eliminated_cycles']
                ],
                'top_k': top_k,
                'date_range': {
                    'since': since,
                    'until': until
                }
            }
            json.dump(serializable, f, indent=2)

        # Save report
        report_file = os.path.join(
            output_dir,
            f"{results['project_name']}_validation_report.txt"
        )
        with open(report_file, 'w') as f:
            f.write(validation_results['report'])

        click.echo(f"\n✓ Validation complete!")
        click.echo(f"Results saved to: {output_dir}")
        click.echo(f"  - {os.path.basename(validation_file)}")
        click.echo(f"  - {os.path.basename(report_file)}")

    except Exception as e:
        click.echo(f"Error during validation: {e}", err=True)
        if cfg.get('verbose', True):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--config', default='config.yaml', type=click.Path(exists=True),
              help='Configuration file')
def info(config):
    """
    Display system information and configuration.
    """
    with open(config) as f:
        cfg = yaml.safe_load(f)

    click.echo("\n" + "="*80)
    click.echo("SEMANTIC CYCLE PRIORITIZATION SYSTEM")
    click.echo("="*80 + "\n")

    click.echo("Configuration:")
    click.echo(f"  Embedding Model:     {cfg.get('embedding_model', 'N/A')}")
    click.echo(f"  Semantic Weight:     {cfg.get('semantic_weight', 'N/A')}")
    click.echo(f"  Structural Weight:   {cfg.get('structural_weight', 'N/A')}")
    click.echo(f"  Dynamic Weight:      {cfg.get('dynamic_weight', 'N/A')}")
    click.echo(f"  Cache Directory:     {cfg.get('cache_dir', 'N/A')}")
    click.echo(f"  Output Directory:    {cfg.get('output_dir', 'N/A')}")

    click.echo("\nSystem Requirements:")

    # Check dependencies
    deps = {
        'tree-sitter': None,
        'javalang': None,
        'igraph': None,
        'sentence-transformers': None,
        'pandas': None,
        'numpy': None,
        'scipy': None
    }

    for dep in deps:
        try:
            __import__(dep.replace('-', '_'))
            status = "✓ Installed"
        except ImportError:
            status = "✗ Missing"
        click.echo(f"  {dep:25} {status}")


@cli.command()
@click.option('--project', required=True, type=click.Path(exists=True),
              help='Path to Java project directory')
@click.option('--output', default='data/results/', type=click.Path(),
              help='Output directory for statistics')
def stats(project, output):
    """
    Compute and display project statistics without full analysis.

    Quickly analyze project structure and basic metrics.
    """
    click.echo(f"Computing statistics for: {project}\n")

    from src.parser.java_parser import JavaParser
    import glob

    parser = JavaParser()

    # Find Java files
    pattern = os.path.join(project, '**', '*.java')
    java_files = glob.glob(pattern, recursive=True)

    click.echo(f"Java Files:     {len(java_files)}")

    # Parse all files
    classes = []
    packages = set()

    for file_path in java_files:
        try:
            parsed = parser.parse_file(file_path)
            if parsed:
                classes.append(parsed)
                if parsed.package_name:
                    packages.add(parsed.package_name)
        except:
            pass

    click.echo(f"Parsed Classes: {len(classes)}")
    click.echo(f"Packages:       {len(packages)}")

    # Method and field counts
    total_methods = sum(len(c.methods) for c in classes)
    total_fields = sum(len(c.fields) for c in classes)

    click.echo(f"Methods:        {total_methods}")
    click.echo(f"Fields:         {total_fields}")

    # Averages
    if classes:
        click.echo(f"\nAverages:")
        click.echo(f"  Methods per class:  {total_methods / len(classes):.1f}")
        click.echo(f"  Fields per class:   {total_fields / len(classes):.1f}")


@cli.command()
@click.option('--project', required=True, type=click.Path(exists=True),
              help='Path to Java project directory')
@click.option('--config', default='config.yaml', type=click.Path(exists=True),
              help='Configuration file')
@click.option('--output', default=None, type=click.Path(),
              help='Output directory')
@click.option('--at-commit', default=None, type=str,
              help='Analyze project at specific commit (hash or date YYYY-MM-DD)')
@click.option('--min-scc-size', default=10, type=int,
              help='Minimum SCC size to decompose (default: 10)')
@click.option('--max-cycles', default=500, type=int,
              help='Maximum simple cycles to find per SCC (default: 500)')
def decompose(project, config, output, at_commit, min_scc_size, max_cycles):
    """
    Decompose large SCCs into simple cycles.

    This command analyzes the project, finds large Strongly Connected Components (SCCs),
    and breaks them down into elementary simple cycles for easier understanding.

    Example:
        python run_analysis.py decompose --project /path/to/java/project
        python run_analysis.py decompose --project /path/to/java/project --at-commit 2015-01-01
        python run_analysis.py decompose --project /path/to/java/project --min-scc-size 20
    """
    # Load configuration
    with open(config) as f:
        cfg = yaml.safe_load(f)

    if output:
        cfg['output_dir'] = output

    # Handle --at-commit option
    git_root = None
    worktree_path = None
    project_to_analyze = project

    if at_commit:
        current_path = os.path.abspath(project)
        for _ in range(5):
            if os.path.exists(os.path.join(current_path, '.git')):
                git_root = current_path
                break
            parent = os.path.dirname(current_path)
            if parent == current_path:
                break
            current_path = parent

        if not git_root:
            click.echo("Error: --at-commit requires a git repository.", err=True)
            sys.exit(1)

        click.echo(f"Checking out project at commit/date: {at_commit}...")
        commit_hash = _resolve_commit_ref(git_root, at_commit)

        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ci', commit_hash],
            cwd=git_root, capture_output=True, text=True, check=True
        )
        commit_date = result.stdout.strip().split(' ')[0]
        click.echo(f"  Resolved to commit: {commit_hash[:8]} (date: {commit_date})\n")

        git_root_abs = os.path.abspath(git_root)
        project_abs = os.path.abspath(project)
        if project_abs.startswith(git_root_abs):
            project_subdir = os.path.relpath(project_abs, git_root_abs)
            if project_subdir == '.':
                project_subdir = ''
        else:
            project_subdir = ''

        worktree_path = _checkout_at_commit(git_root, project_subdir, commit_hash)
        project_to_analyze = worktree_path

    try:
        # Run basic analysis first
        click.echo("Step 1: Analyzing project for SCCs...")
        pipeline = CyclePrioritizationPipeline(cfg)
        results = pipeline.analyze_project(project_to_analyze)

        num_sccs = len(results.get('cycles', []))
        click.echo(f"  Found {num_sccs} SCCs (Strongly Connected Components)")

        # Get SCCs
        sccs = results.get('cycles', [])

        # Find large SCCs
        large_sccs = [scc for scc in sccs if len(scc.get('cycle_classes', [])) >= min_scc_size]

        click.echo(f"  {len(large_sccs)} SCCs have >= {min_scc_size} classes\n")

        if not large_sccs:
            click.echo("No large SCCs found. Try reducing --min-scc-size.")
            return

        # Import decomposer
        from src.graph.cycle_decomposer import CycleDecomposer

        # Get the graph (DependencyGraph object contains igraph.Graph)
        dependency_graph = results['dependency_graph']
        graph = dependency_graph.graph

        decomposer = CycleDecomposer(graph)

        # Decompose each large SCC
        click.echo("Step 2: Decomposing large SCCs into simple cycles...")
        all_subcycles = {}

        for i, scc in enumerate(large_sccs):
            scc_id = scc['cycle_id']
            scc_classes = scc['cycle_classes']
            scc_size = len(scc_classes)

            click.echo(f"\n  Decomposing {scc_id} ({scc_size} classes)...")

            # Get node indices for this SCC
            scc_nodes = []
            for class_name in scc_classes:
                try:
                    node_idx = graph.vs.find(name=class_name).index
                    scc_nodes.append(node_idx)
                except:
                    pass

            if not scc_nodes:
                click.echo(f"    Warning: Could not find nodes for {scc_id}")
                continue

            # Decompose
            simple_cycles = decomposer.decompose_scc(scc_nodes, max_cycles=max_cycles)

            click.echo(f"    Found {len(simple_cycles)} simple cycles")

            # Analyze overlap
            if simple_cycles:
                overlap_stats = decomposer.analyze_cycle_overlap(simple_cycles)
                click.echo(f"    Avg cycle size: {overlap_stats['avg_cycle_size']:.1f} classes")
                click.echo(f"    Size range: {overlap_stats['min_cycle_size']}-{overlap_stats['max_cycle_size']} classes")

                # Get embeddings and structural metrics from results
                embeddings = results.get('embeddings', {})
                structural_metrics_df = results.get('structural_metrics')

                # Convert DataFrame to dictionary format for compute_subcycle_metrics
                structural_metrics = {}
                if structural_metrics_df is not None and not structural_metrics_df.empty:
                    for _, row in structural_metrics_df.iterrows():
                        class_name = row['class_name']
                        structural_metrics[class_name] = {
                            'cbo': row.get('CBO', 0),
                            'fan_out': row.get('Ce', 0),  # Efferent coupling
                            'fan_in': row.get('Ca', 0),    # Afferent coupling
                            'instability': row.get('Instability', 0)
                        }

                # Compute metrics for subcycles (semantic sim, anomaly score, etc.)
                ranked_subcycles = decomposer.compute_subcycle_metrics(
                    simple_cycles,
                    graph,
                    embeddings=embeddings,
                    structural_metrics=structural_metrics
                )

                # Show anomaly score range if computed
                if ranked_subcycles and ranked_subcycles[0].get('anomaly_score') is not None:
                    scores = [c['anomaly_score'] for c in ranked_subcycles if c['anomaly_score'] is not None]
                    if scores:
                        click.echo(f"    Anomaly scores: {min(scores):.3f} - {max(scores):.3f}")

                all_subcycles[scc_id] = {
                    'original_scc': scc,
                    'num_subcycles': len(simple_cycles),
                    'subcycles': ranked_subcycles,
                    'stats': overlap_stats
                }

        # Save results
        click.echo("\nStep 3: Saving decomposition results...")
        output_dir = cfg.get('output_dir', 'data/results/')
        os.makedirs(output_dir, exist_ok=True)

        import json

        # Prepare filename
        project_name = results['project_name']
        if at_commit:
            filename_base = f"{project_name}_{commit_date.replace('-', '')}"
        else:
            filename_base = f"{project_name}_current"

        decomp_file = os.path.join(output_dir, f"{filename_base}_decomposed.json")

        with open(decomp_file, 'w', encoding='utf-8') as f:
            serializable = {
                'project': project_name,
                'total_sccs': num_sccs,
                'decomposed_sccs': len(all_subcycles),
                'min_scc_size': min_scc_size,
                'max_cycles_per_scc': max_cycles,
                'sccs': all_subcycles
            }
            # Convert sets to lists for JSON serialization
            serializable = convert_to_serializable(serializable)
            json.dump(serializable, f, indent=2)

        # Create summary report
        report_file = os.path.join(output_dir, f"{filename_base}_decomposition_summary.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SCC DECOMPOSITION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Project: {project_name}\n")
            f.write(f"Total SCCs: {num_sccs}\n")
            f.write(f"Large SCCs (>= {min_scc_size} classes): {len(large_sccs)}\n")
            f.write(f"Decomposed: {len(all_subcycles)}\n\n")

            for scc_id, data in all_subcycles.items():
                f.write("-" * 80 + "\n")
                f.write(f"{scc_id}: {data['original_scc']['cycle_classes'][0]}, ...\n")
                f.write(f"  Original SCC size: {len(data['original_scc']['cycle_classes'])} classes\n")
                f.write(f"  Simple cycles found: {data['num_subcycles']}\n")
                f.write(f"  Avg subcycle size: {data['stats']['avg_cycle_size']:.1f} classes\n")
                f.write(f"  Size range: {data['stats']['min_cycle_size']}-{data['stats']['max_cycle_size']}\n")

                # Add anomaly score statistics if available
                anomaly_scores = [c.get('anomaly_score') for c in data['subcycles'] if c.get('anomaly_score') is not None]
                if anomaly_scores:
                    f.write(f"  Anomaly score range: {min(anomaly_scores):.3f}-{max(anomaly_scores):.3f}\n")

                # Show top 5 subcycles by anomaly score
                f.write(f"\n  Top 5 subcycles by anomaly score:\n")
                for i, subcycle in enumerate(data['subcycles'][:5], 1):
                    classes_str = ' → '.join(subcycle['classes'][:5])
                    if len(subcycle['classes']) > 5:
                        classes_str += ' → ...'

                    # Include semantic similarity and anomaly scores if available
                    sem_sim = subcycle.get('semantic_similarity')
                    anom = subcycle.get('anomaly_score')
                    struct_risk = subcycle.get('structural_risk')

                    if anom is not None and sem_sim is not None:
                        f.write(f"    {i}. Size {subcycle['size']}, Sem:{sem_sim:.3f}, Anomaly:{anom:.3f}: {classes_str}\n")
                    else:
                        f.write(f"    {i}. Size {subcycle['size']}: {classes_str}\n")

                f.write("\n")

        click.echo(f"\n✓ Decomposition complete!")
        click.echo(f"Results saved to: {output_dir}")
        click.echo(f"  - {os.path.basename(decomp_file)}")
        click.echo(f"  - {os.path.basename(report_file)}")

    except Exception as e:
        click.echo(f"Error during decomposition: {e}", err=True)
        if cfg.get('verbose', True):
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        if worktree_path:
            click.echo("\nCleaning up temporary worktree...")
            _cleanup_worktree(git_root, worktree_path)


if __name__ == '__main__':
    cli()
