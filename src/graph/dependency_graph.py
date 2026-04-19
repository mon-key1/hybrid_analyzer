"""Dependency graph construction and analysis using igraph."""

from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd

try:
    import igraph as ig
except ImportError:
    print("Warning: igraph not installed. Install with: pip install igraph")
    ig = None


class DependencyGraph:
    """Dependency graph using igraph for performance."""

    def __init__(self):
        """Initialize an empty dependency graph."""
        if ig is None:
            raise ImportError("igraph is required. Install with: pip install igraph")

        self.graph = ig.Graph(directed=True)
        self.class_to_vertex: Dict[str, int] = {}
        self.vertex_to_class: Dict[int, str] = {}
        self._next_vertex_id = 0

    def add_class(self, class_name: str, package: str, metadata: Optional[Dict] = None):
        """
        Add a vertex for a class with metadata.

        Args:
            class_name: Fully qualified class name
            package: Package name
            metadata: Additional metadata (parsed_class, methods, fields, etc.)
        """
        if class_name in self.class_to_vertex:
            # Update metadata if vertex exists
            vertex_id = self.class_to_vertex[class_name]
            if metadata:
                for key, value in metadata.items():
                    self.graph.vs[vertex_id][key] = value
            return

        # Add new vertex
        vertex_id = self._next_vertex_id
        self.graph.add_vertex(name=class_name)

        # Set attributes
        self.graph.vs[vertex_id]['class_name'] = class_name
        self.graph.vs[vertex_id]['package'] = package

        if metadata:
            for key, value in metadata.items():
                self.graph.vs[vertex_id][key] = value

        # Update indices
        self.class_to_vertex[class_name] = vertex_id
        self.vertex_to_class[vertex_id] = class_name
        self._next_vertex_id += 1

    def add_dependency(self, source: str, target: str, edge_type: str = 'uses', weight: float = 1.0):
        """
        Add a directed dependency edge.

        Args:
            source: Source class name
            target: Target class name
            edge_type: Type of dependency (uses, extends, implements, etc.)
            weight: Edge weight (default 1.0)
        """
        # Ensure both vertices exist
        if source not in self.class_to_vertex:
            # Extract package from class name
            package = '.'.join(source.split('.')[:-1]) if '.' in source else ''
            self.add_class(source, package)

        if target not in self.class_to_vertex:
            package = '.'.join(target.split('.')[:-1]) if '.' in target else ''
            self.add_class(target, package)

        source_id = self.class_to_vertex[source]
        target_id = self.class_to_vertex[target]

        # Add edge
        self.graph.add_edge(source_id, target_id, type=edge_type, weight=weight)

    def compute_structural_metrics(self) -> pd.DataFrame:
        """
        Compute structural metrics for all classes.

        Returns:
            DataFrame with columns:
            - class_name
            - package
            - CBO (Coupling Between Objects)
            - Ca (Afferent Coupling)
            - Ce (Efferent Coupling)
            - Instability (Ce / (Ca + Ce))
            - in_degree
            - out_degree
        """
        metrics = []

        for vertex_id in range(self.graph.vcount()):
            class_name = self.vertex_to_class[vertex_id]
            package = self.graph.vs[vertex_id]['package'] if 'package' in self.graph.vs[vertex_id].attributes() else ''

            # Compute metrics
            in_degree = self.graph.indegree(vertex_id)
            out_degree = self.graph.outdegree(vertex_id)

            # Ca = Afferent coupling (classes that depend on this class)
            ca = in_degree

            # Ce = Efferent coupling (classes this class depends on)
            ce = out_degree

            # CBO = Coupling Between Objects (total unique dependencies)
            neighbors_in = set(self.graph.neighbors(vertex_id, mode='in'))
            neighbors_out = set(self.graph.neighbors(vertex_id, mode='out'))
            cbo = len(neighbors_in | neighbors_out)

            # Instability = Ce / (Ca + Ce)
            instability = ce / (ca + ce) if (ca + ce) > 0 else 0.0

            metrics.append({
                'class_name': class_name,
                'package': package,
                'CBO': cbo,
                'Ca': ca,
                'Ce': ce,
                'Instability': instability,
                'in_degree': in_degree,
                'out_degree': out_degree
            })

        return pd.DataFrame(metrics)

    def get_subgraph_by_package(self, package: str) -> 'DependencyGraph':
        """
        Extract subgraph for a specific package.

        Args:
            package: Package name

        Returns:
            New DependencyGraph containing only classes from the package
        """
        # Find vertices in the package
        vertices_to_keep = []
        for vertex_id in range(self.graph.vcount()):
            vertex_package = self.graph.vs[vertex_id]['package'] if 'package' in self.graph.vs[vertex_id].attributes() else ''
            if vertex_package.startswith(package):
                vertices_to_keep.append(vertex_id)

        # Create subgraph
        subgraph_ig = self.graph.subgraph(vertices_to_keep)

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

    def get_vertex_id(self, class_name: str) -> Optional[int]:
        """Get vertex ID for a class name."""
        return self.class_to_vertex.get(class_name)

    def get_class_name(self, vertex_id: int) -> Optional[str]:
        """Get class name for a vertex ID."""
        return self.vertex_to_class.get(vertex_id)

    def get_neighbors(self, class_name: str, mode: str = 'all') -> List[str]:
        """
        Get neighbors of a class.

        Args:
            class_name: Class name
            mode: 'in', 'out', or 'all'

        Returns:
            List of neighbor class names
        """
        vertex_id = self.class_to_vertex.get(class_name)
        if vertex_id is None:
            return []

        neighbor_ids = self.graph.neighbors(vertex_id, mode=mode)
        return [self.vertex_to_class[nid] for nid in neighbor_ids if nid in self.vertex_to_class]

    def get_edges_between(self, class_names: List[str]) -> List[Tuple[str, str, str]]:
        """
        Get all edges between a set of classes.

        Args:
            class_names: List of class names

        Returns:
            List of (source, target, edge_type) tuples
        """
        vertex_ids = [self.class_to_vertex[name] for name in class_names
                     if name in self.class_to_vertex]

        edges = []
        for edge in self.graph.es:
            if edge.source in vertex_ids and edge.target in vertex_ids:
                source = self.vertex_to_class[edge.source]
                target = self.vertex_to_class[edge.target]
                edge_type = edge['type'] if 'type' in edge.attributes() else 'uses'
                edges.append((source, target, edge_type))

        return edges

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get basic graph statistics."""
        return {
            'num_classes': self.graph.vcount(),
            'num_dependencies': self.graph.ecount(),
            'num_packages': len(set(v['package'] if 'package' in v.attributes() else '' for v in self.graph.vs)),
            'density': self.graph.density(),
            'is_connected': self.graph.is_connected(mode='weak')
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary format."""
        vertices = []
        for vertex_id in range(self.graph.vcount()):
            vertex_data = {
                'id': vertex_id,
                'class_name': self.vertex_to_class[vertex_id],
                'package': self.graph.vs[vertex_id]['package'] if 'package' in self.graph.vs[vertex_id].attributes() else ''
            }
            vertices.append(vertex_data)

        edges = []
        for edge in self.graph.es:
            edge_data = {
                'source': self.vertex_to_class[edge.source],
                'target': self.vertex_to_class[edge.target],
                'type': edge['type'] if 'type' in edge.attributes() else 'uses',
                'weight': edge['weight'] if 'weight' in edge.attributes() else 1.0
            }
            edges.append(edge_data)

        return {
            'vertices': vertices,
            'edges': edges,
            'stats': self.get_graph_stats()
        }