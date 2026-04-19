"""Decompose large SCCs into simple cycles."""

from typing import List, Dict, Set, Tuple
import igraph as ig
from collections import defaultdict


class CycleDecomposer:
    """Decompose Strongly Connected Components into elementary cycles."""

    def __init__(self, graph: ig.Graph):
        """
        Initialize decomposer.

        Args:
            graph: Directed graph
        """
        self.graph = graph

    def decompose_scc(self, scc_nodes: List[int], max_cycles: int = 1000) -> List[List[int]]:
        """
        Find all simple cycles within an SCC.

        Args:
            scc_nodes: List of node indices forming the SCC
            max_cycles: Maximum number of cycles to find (prevent explosion)

        Returns:
            List of simple cycles (each cycle is a list of node indices)
        """
        # Extract subgraph for this SCC
        subgraph = self.graph.subgraph(scc_nodes)

        # Find all simple cycles in the subgraph
        # Note: This uses Johnson's algorithm internally
        try:
            # igraph has a method for finding simple cycles but it's limited
            # We'll use a custom implementation for better control
            cycles = self._find_simple_cycles_johnson(subgraph, max_cycles=max_cycles)
        except Exception as e:
            print(f"Warning: Could not decompose SCC with {len(scc_nodes)} nodes: {e}")
            cycles = []

        # Map subgraph indices back to original graph indices
        mapped_cycles = []
        for cycle in cycles:
            mapped_cycle = [scc_nodes[i] for i in cycle]
            mapped_cycles.append(mapped_cycle)

        return mapped_cycles

    def _find_simple_cycles_johnson(self, graph: ig.Graph, max_cycles: int = 1000) -> List[List[int]]:
        """
        Find all simple cycles using Johnson's algorithm.

        This is a simplified implementation that finds elementary cycles.

        Args:
            graph: Directed graph
            max_cycles: Maximum cycles to find

        Returns:
            List of cycles (node index lists)
        """
        n = graph.vcount()
        cycles = []

        # Build adjacency list
        adj = defaultdict(set)
        for edge in graph.es:
            adj[edge.source].add(edge.target)

        # For small graphs, use exhaustive search
        if n <= 10:
            return self._find_cycles_small_graph(graph, max_cycles)

        # For larger graphs, find minimal cycles using DFS
        visited_global = set()

        for start_node in range(n):
            if start_node in visited_global:
                continue

            # Find cycles starting from this node
            stack = [(start_node, [start_node], {start_node})]

            while stack and len(cycles) < max_cycles:
                node, path, visited = stack.pop()

                for neighbor in adj[node]:
                    if neighbor == start_node and len(path) > 1:
                        # Found a cycle back to start
                        cycles.append(path[:])
                    elif neighbor not in visited and neighbor >= start_node:
                        # Continue search (only visit nodes >= start to avoid duplicates)
                        new_path = path + [neighbor]
                        new_visited = visited | {neighbor}
                        stack.append((neighbor, new_path, new_visited))

            visited_global.add(start_node)

        return cycles[:max_cycles]

    def _find_cycles_small_graph(self, graph: ig.Graph, max_cycles: int) -> List[List[int]]:
        """
        Find cycles in small graphs using simpler algorithm.

        Args:
            graph: Small directed graph
            max_cycles: Max cycles to find

        Returns:
            List of cycles
        """
        n = graph.vcount()
        cycles = []

        # Build adjacency
        adj = defaultdict(set)
        for edge in graph.es:
            adj[edge.source].add(edge.target)

        # DFS from each node
        for start in range(n):
            stack = [(start, [start])]

            while stack and len(cycles) < max_cycles:
                node, path = stack.pop()

                for neighbor in adj[node]:
                    if neighbor == start and len(path) >= 2:
                        # Found cycle
                        cycles.append(path[:])
                    elif neighbor not in path and len(path) < 20:  # Limit path length
                        stack.append((neighbor, path + [neighbor]))

        # Remove duplicates (cycles that are rotations of each other)
        unique_cycles = self._remove_duplicate_cycles(cycles)

        return unique_cycles[:max_cycles]

    def _remove_duplicate_cycles(self, cycles: List[List[int]]) -> List[List[int]]:
        """
        Remove duplicate cycles (rotations and reverses).

        Args:
            cycles: List of cycles

        Returns:
            Unique cycles only
        """
        seen = set()
        unique = []

        for cycle in cycles:
            # Normalize: smallest element first
            min_idx = cycle.index(min(cycle))
            normalized = cycle[min_idx:] + cycle[:min_idx]

            # Convert to tuple for hashing
            cycle_tuple = tuple(normalized)

            if cycle_tuple not in seen:
                seen.add(cycle_tuple)
                unique.append(cycle)

        return unique

    def decompose_all_sccs(self, sccs: List[List[int]],
                          min_size: int = 3,
                          max_cycles_per_scc: int = 500) -> Dict[int, List[List[int]]]:
        """
        Decompose all SCCs that are larger than min_size.

        Args:
            sccs: List of SCCs (each is a list of node indices)
            min_size: Only decompose SCCs with at least this many nodes
            max_cycles_per_scc: Max cycles to find per SCC

        Returns:
            Dictionary mapping SCC index to list of simple cycles
        """
        decomposed = {}

        for i, scc in enumerate(sccs):
            if len(scc) >= min_size:
                print(f"Decomposing SCC {i} ({len(scc)} nodes)...")
                cycles = self.decompose_scc(scc, max_cycles=max_cycles_per_scc)
                if cycles:
                    decomposed[i] = cycles
                    print(f"  Found {len(cycles)} simple cycles")
                else:
                    print(f"  No simple cycles found (or too complex)")

        return decomposed

    def analyze_cycle_overlap(self, cycles: List[List[int]]) -> Dict:
        """
        Analyze how cycles share nodes.

        Args:
            cycles: List of cycles

        Returns:
            Dictionary with overlap statistics
        """
        # Count how many cycles each node appears in
        node_frequency = defaultdict(int)
        for cycle in cycles:
            for node in cycle:
                node_frequency[node] += 1

        # Find most shared nodes
        sorted_nodes = sorted(node_frequency.items(), key=lambda x: x[1], reverse=True)

        # Count cycle sizes
        cycle_sizes = [len(c) for c in cycles]

        return {
            'total_cycles': len(cycles),
            'avg_cycle_size': sum(cycle_sizes) / len(cycles) if cycles else 0,
            'min_cycle_size': min(cycle_sizes) if cycles else 0,
            'max_cycle_size': max(cycle_sizes) if cycles else 0,
            'most_shared_nodes': sorted_nodes[:10],  # Top 10
            'unique_nodes': len(node_frequency)
        }

    def get_minimal_feedback_set(self, cycles: List[List[int]]) -> Set[int]:
        """
        Find minimal set of nodes whose removal breaks all cycles.

        This is the Feedback Vertex Set problem (NP-hard).
        We use a greedy approximation.

        Args:
            cycles: List of cycles

        Returns:
            Set of node indices to remove
        """
        # Greedy: repeatedly remove the node that appears in most cycles
        remaining_cycles = [set(c) for c in cycles]
        feedback_set = set()

        while remaining_cycles:
            # Count node frequencies
            node_count = defaultdict(int)
            for cycle_set in remaining_cycles:
                for node in cycle_set:
                    node_count[node] += 1

            if not node_count:
                break

            # Remove most frequent node
            most_common_node = max(node_count.items(), key=lambda x: x[1])[0]
            feedback_set.add(most_common_node)

            # Remove all cycles containing this node
            remaining_cycles = [
                cycle_set for cycle_set in remaining_cycles
                if most_common_node not in cycle_set
            ]

        return feedback_set

    def compute_subcycle_metrics(self, cycles: List[List[int]],
                                 graph: ig.Graph,
                                 embeddings: Dict = None,
                                 structural_metrics: Dict = None) -> List[Dict]:
        """
        Compute semantic and structural metrics for subcycles.

        Args:
            cycles: List of simple cycles (node indices)
            graph: Original graph
            embeddings: Dictionary mapping class names to embedding vectors
            structural_metrics: Dictionary mapping class names to structural metrics

        Returns:
            List of subcycles with full metrics (semantic_sim, structural_risk, anomaly_score)
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        ranked_cycles = []

        for i, cycle in enumerate(cycles):
            cycle_classes = [graph.vs[node]['name'] for node in cycle]
            cycle_size = len(cycle)

            # Basic structural metrics
            edge_count = 0
            for j in range(len(cycle)):
                curr = cycle[j]
                next_node = cycle[(j + 1) % len(cycle)]
                if graph.are_connected(curr, next_node):
                    edge_count += 1

            edge_density = edge_count / (cycle_size * (cycle_size - 1)) if cycle_size > 1 else 0

            # Compute semantic similarity if embeddings available
            semantic_sim = None
            if embeddings:
                cycle_embeddings = []
                for class_name in cycle_classes:
                    if class_name in embeddings:
                        cycle_embeddings.append(embeddings[class_name])

                if len(cycle_embeddings) >= 2:
                    # Compute average pairwise similarity
                    sims = []
                    for j in range(len(cycle_embeddings)):
                        for k in range(j + 1, len(cycle_embeddings)):
                            sim = cosine_similarity(
                                [cycle_embeddings[j]],
                                [cycle_embeddings[k]]
                            )[0][0]
                            sims.append(sim)
                    semantic_sim = np.mean(sims) if sims else None

            # Compute structural risk if metrics available
            structural_risk = None
            if structural_metrics:
                # Average CBO, WMC, etc. for classes in cycle
                cbo_values = []
                fan_out_values = []

                for class_name in cycle_classes:
                    if class_name in structural_metrics:
                        metrics = structural_metrics[class_name]
                        if 'cbo' in metrics:
                            cbo_values.append(metrics['cbo'])
                        if 'fan_out' in metrics:
                            fan_out_values.append(metrics['fan_out'])

                if cbo_values:
                    avg_cbo = np.mean(cbo_values)
                    # Normalize to [0, 1] range (assuming max CBO ~50)
                    structural_risk = min(avg_cbo / 50.0, 1.0)

            # Compute anomaly score (combine semantic and structural)
            anomaly_score = None
            if semantic_sim is not None and structural_risk is not None:
                # Lower semantic sim + higher structural risk = higher anomaly
                # Invert semantic_sim so lower similarity gives higher score
                semantic_component = 1.0 - semantic_sim
                # Combine (weights: 0.6 semantic, 0.4 structural)
                anomaly_score = 0.6 * semantic_component + 0.4 * structural_risk

            cycle_data = {
                'subcycle_id': f'subcycle_{i:04d}',
                'nodes': cycle,
                'classes': cycle_classes,
                'size': cycle_size,
                'edge_count': edge_count,
                'edge_density': edge_density,
                'semantic_similarity': semantic_sim,
                'structural_risk': structural_risk,
                'anomaly_score': anomaly_score
            }

            ranked_cycles.append(cycle_data)

        # Sort by anomaly score (highest first) if available, otherwise by size
        if any(c.get('anomaly_score') is not None for c in ranked_cycles):
            ranked_cycles.sort(key=lambda x: (x.get('anomaly_score') or 0), reverse=True)
        else:
            ranked_cycles.sort(key=lambda x: (x['size'], -x['edge_density']))

        return ranked_cycles

    def prioritize_subcycles(self, cycles: List[List[int]],
                            graph: ig.Graph,
                            class_metrics: Dict = None) -> List[Dict]:
        """
        Rank subcycles by various criteria.

        Args:
            cycles: List of cycles (node indices)
            graph: Original graph
            class_metrics: Optional metrics for classes

        Returns:
            List of cycle dictionaries with priority scores
        """
        ranked_cycles = []

        for i, cycle in enumerate(cycles):
            cycle_classes = [graph.vs[node]['name'] for node in cycle]

            # Compute cycle metrics
            cycle_size = len(cycle)

            # Count edges in cycle
            edge_count = 0
            for j in range(len(cycle)):
                curr = cycle[j]
                next_node = cycle[(j + 1) % len(cycle)]
                if graph.are_connected(curr, next_node):
                    edge_count += 1

            # Structural risk: how tightly connected
            edge_density = edge_count / (cycle_size * (cycle_size - 1)) if cycle_size > 1 else 0

            cycle_data = {
                'subcycle_id': f'subcycle_{i:04d}',
                'nodes': cycle,
                'classes': cycle_classes,
                'size': cycle_size,
                'edge_count': edge_count,
                'edge_density': edge_density
            }

            ranked_cycles.append(cycle_data)

        # Sort by size (smaller cycles are often easier to fix)
        ranked_cycles.sort(key=lambda x: (x['size'], -x['edge_density']))

        return ranked_cycles
