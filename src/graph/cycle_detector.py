"""Cycle detection using Tarjan's SCC algorithm."""

from typing import List, Dict, Tuple, Set
from .dependency_graph import DependencyGraph


class CycleDetector:
    """Detect cycles using Tarjan's algorithm for strongly connected components."""

    def __init__(self, graph: DependencyGraph):
        """
        Initialize cycle detector.

        Args:
            graph: DependencyGraph to analyze
        """
        self.graph = graph

    def find_strongly_connected_components(self) -> List[List[str]]:
        """
        Find all SCCs (cycles) in the graph.

        Returns:
            List of SCCs, each SCC is a list of class names.
            Filters out SCCs with size=1 (single nodes without self-loops).
        """
        # Use igraph's built-in SCC detection
        scc_components = self.graph.graph.components(mode='strong')

        cycles = []
        for component in scc_components:
            # Filter out single-node components without self-loops
            if len(component) > 1:
                class_names = [self.graph.vertex_to_class[vid] for vid in component
                             if vid in self.graph.vertex_to_class]
                if class_names:
                    cycles.append(class_names)
            elif len(component) == 1:
                # Check for self-loop
                vertex_id = component[0]
                if self.graph.graph.are_connected(vertex_id, vertex_id):
                    class_name = self.graph.vertex_to_class.get(vertex_id)
                    if class_name:
                        cycles.append([class_name])

        return cycles

    def extract_cycle_edges(self, cycle_classes: List[str]) -> List[Tuple[str, str, str]]:
        """
        Extract all edges within a cycle.

        Args:
            cycle_classes: List of class names in the cycle

        Returns:
            List of (source, target, edge_type) tuples
        """
        return self.graph.get_edges_between(cycle_classes)

    def classify_cycle(self, cycle: List[str]) -> Dict:
        """
        Classify cycle properties.

        Args:
            cycle: List of class names in the cycle

        Returns:
            Dictionary with:
            - type: 'cross-package' or 'intra-package'
            - size: number of classes
            - edge_count: number of dependencies
            - packages: set of packages involved
        """
        # Get packages for each class
        packages = set()
        for class_name in cycle:
            vertex_id = self.graph.class_to_vertex.get(class_name)
            if vertex_id is not None:
                # igraph.Vertex doesn't have .get(), use indexing
                vertex = self.graph.graph.vs[vertex_id]
                package = vertex['package'] if 'package' in vertex.attributes() else ''
                if package:
                    packages.add(package)

        # Get edges
        edges = self.extract_cycle_edges(cycle)

        # Determine type
        cycle_type = 'cross-package' if len(packages) > 1 else 'intra-package'

        return {
            'type': cycle_type,
            'size': len(cycle),
            'edge_count': len(edges),
            'packages': packages
        }

    def find_all_cycles_with_classification(self) -> List[Dict]:
        """
        Find all cycles with their classifications.

        Returns:
            List of dictionaries, each containing:
            - cycle_id: str
            - cycle_classes: List[str]
            - cycle_edges: List[Tuple[str, str, str]]
            - classification: Dict (from classify_cycle)
        """
        sccs = self.find_strongly_connected_components()

        cycles_with_info = []
        for i, cycle_classes in enumerate(sccs):
            cycle_id = f"cycle_{i:03d}"
            edges = self.extract_cycle_edges(cycle_classes)
            classification = self.classify_cycle(cycle_classes)

            cycles_with_info.append({
                'cycle_id': cycle_id,
                'cycle_classes': cycle_classes,
                'cycle_edges': edges,
                'classification': classification
            })

        return cycles_with_info

    def get_cycle_subgraph(self, cycle_classes: List[str]) -> DependencyGraph:
        """
        Extract a subgraph containing only the cycle.

        Args:
            cycle_classes: List of class names in the cycle

        Returns:
            DependencyGraph containing only the cycle
        """
        # Get vertex IDs
        vertex_ids = [self.graph.class_to_vertex[cls] for cls in cycle_classes
                     if cls in self.graph.class_to_vertex]

        # Create subgraph
        subgraph_ig = self.graph.graph.subgraph(vertex_ids)

        # Create new DependencyGraph
        subgraph = DependencyGraph()
        subgraph.graph = subgraph_ig

        # Rebuild indices
        for i, vertex in enumerate(subgraph.graph.vs):
            class_name = vertex['class_name']
            subgraph.class_to_vertex[class_name] = i
            subgraph.vertex_to_class[i] = class_name

        subgraph._next_vertex_id = len(subgraph.graph.vs)

        return subgraph

    def find_breaking_edges(self, cycle_classes: List[str]) -> List[Tuple[str, str, float]]:
        """
        Find edges that, if removed, would break the cycle.

        Uses a simple heuristic: edges with highest betweenness centrality.

        Args:
            cycle_classes: List of class names in the cycle

        Returns:
            List of (source, target, importance_score) tuples, sorted by score
        """
        cycle_subgraph = self.get_cycle_subgraph(cycle_classes)

        # Compute edge betweenness
        try:
            edge_betweenness = cycle_subgraph.graph.edge_betweenness()
        except:
            # Fallback: use degree-based heuristic
            edge_betweenness = [1.0] * cycle_subgraph.graph.ecount()

        # Create list of edges with scores
        breaking_edges = []
        for i, edge in enumerate(cycle_subgraph.graph.es):
            source = cycle_subgraph.vertex_to_class.get(edge.source, '')
            target = cycle_subgraph.vertex_to_class.get(edge.target, '')
            score = edge_betweenness[i] if i < len(edge_betweenness) else 1.0

            breaking_edges.append((source, target, score))

        # Sort by score (descending)
        breaking_edges.sort(key=lambda x: x[2], reverse=True)

        return breaking_edges

    def get_cycle_statistics(self) -> Dict:
        """
        Get statistics about all cycles in the graph.

        Returns:
            Dictionary with cycle statistics
        """
        cycles = self.find_strongly_connected_components()

        if not cycles:
            return {
                'total_cycles': 0,
                'total_classes_in_cycles': 0,
                'avg_cycle_size': 0.0,
                'max_cycle_size': 0,
                'min_cycle_size': 0
            }

        cycle_sizes = [len(c) for c in cycles]
        classes_in_cycles = set()
        for cycle in cycles:
            classes_in_cycles.update(cycle)

        return {
            'total_cycles': len(cycles),
            'total_classes_in_cycles': len(classes_in_cycles),
            'avg_cycle_size': sum(cycle_sizes) / len(cycle_sizes),
            'max_cycle_size': max(cycle_sizes),
            'min_cycle_size': min(cycle_sizes)
        }