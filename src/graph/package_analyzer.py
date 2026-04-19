"""Package-level dependency analysis and cycle detection."""

from typing import List, Dict, Set, Tuple
from collections import defaultdict
import igraph as ig

from .dependency_graph import DependencyGraph
from .cycle_detector import CycleDetector


class PackageAnalyzer:
    """Analyze dependencies and cycles at package level."""

    def __init__(self, class_graph: DependencyGraph):
        """
        Initialize package analyzer.

        Args:
            class_graph: Class-level dependency graph
        """
        self.class_graph = class_graph
        self.package_graph = None
        self.package_to_classes = defaultdict(set)
        self.class_to_package = {}

    def build_package_graph(self) -> ig.Graph:
        """
        Build package-level dependency graph from class-level graph.

        Returns:
            igraph Graph with packages as nodes
        """
        # Extract package names from class names
        for class_name in self.class_graph.graph.vs['name']:
            package_name = self._extract_package(class_name)
            self.package_to_classes[package_name].add(class_name)
            self.class_to_package[class_name] = package_name

        # Create package nodes
        packages = sorted(self.package_to_classes.keys())
        self.package_graph = ig.Graph(directed=True)
        self.package_graph.add_vertices(len(packages))
        self.package_graph.vs['name'] = packages

        # Build edges between packages
        package_edges = defaultdict(int)  # (from_pkg, to_pkg) -> count

        for edge in self.class_graph.graph.es:
            source_class = self.class_graph.graph.vs[edge.source]['name']
            target_class = self.class_graph.graph.vs[edge.target]['name']

            source_pkg = self.class_to_package[source_class]
            target_pkg = self.class_to_package[target_class]

            # Skip self-loops (dependencies within same package)
            if source_pkg != target_pkg:
                package_edges[(source_pkg, target_pkg)] += 1

        # Add edges to graph
        edges = []
        weights = []
        for (src_pkg, tgt_pkg), count in package_edges.items():
            src_idx = packages.index(src_pkg)
            tgt_idx = packages.index(tgt_pkg)
            edges.append((src_idx, tgt_idx))
            weights.append(count)

        if edges:
            self.package_graph.add_edges(edges)
            self.package_graph.es['weight'] = weights

        return self.package_graph

    def detect_package_cycles(self) -> List[Dict]:
        """
        Detect cycles at package level.

        Returns:
            List of package cycles with metadata
        """
        if self.package_graph is None:
            self.build_package_graph()

        # Use cycle detector
        detector = CycleDetector(self.package_graph)
        sccs = detector.detect_cycles()

        cycles = []
        for i, scc in enumerate(sccs):
            # Get package names
            package_names = [self.package_graph.vs[node]['name'] for node in scc]

            # Count classes involved
            total_classes = sum(len(self.package_to_classes[pkg]) for pkg in package_names)

            # Count inter-package dependencies
            edge_count = self._count_cycle_edges(scc)

            cycle_data = {
                'cycle_id': f'pkg_cycle_{i:03d}',
                'packages': package_names,
                'num_packages': len(package_names),
                'total_classes': total_classes,
                'inter_package_dependencies': edge_count,
                'package_nodes': scc  # Original node indices
            }

            cycles.append(cycle_data)

        return cycles

    def _count_cycle_edges(self, scc: List[int]) -> int:
        """Count edges within a cycle."""
        scc_set = set(scc)
        count = 0
        for node in scc:
            for neighbor in self.package_graph.neighbors(node, mode='out'):
                if neighbor in scc_set:
                    count += 1
        return count

    def _extract_package(self, class_name: str) -> str:
        """
        Extract package name from fully qualified class name.

        Args:
            class_name: e.g., "org.argouml.ui.ActionSave"

        Returns:
            Package name: e.g., "org.argouml.ui"
        """
        parts = class_name.split('.')
        if len(parts) > 1:
            # Return all but last part (which is class name)
            return '.'.join(parts[:-1])
        else:
            # Default package
            return '(default)'

    def get_package_metrics(self) -> Dict[str, Dict]:
        """
        Compute metrics for each package.

        Returns:
            Dictionary mapping package name to metrics
        """
        if self.package_graph is None:
            self.build_package_graph()

        metrics = {}

        for pkg_idx, pkg_name in enumerate(self.package_graph.vs['name']):
            # Class count
            num_classes = len(self.package_to_classes[pkg_name])

            # In/out dependencies
            in_degree = self.package_graph.degree(pkg_idx, mode='in')
            out_degree = self.package_graph.degree(pkg_idx, mode='out')

            # Afferent/Efferent coupling
            afferent_coupling = in_degree  # How many packages depend on this
            efferent_coupling = out_degree  # How many packages this depends on

            # Instability (I = Ce / (Ca + Ce))
            total_coupling = afferent_coupling + efferent_coupling
            if total_coupling > 0:
                instability = efferent_coupling / total_coupling
            else:
                instability = 0.0

            metrics[pkg_name] = {
                'num_classes': num_classes,
                'afferent_coupling': afferent_coupling,
                'efferent_coupling': efferent_coupling,
                'instability': instability,
                'in_degree': in_degree,
                'out_degree': out_degree
            }

        return metrics

    def compute_package_cycle_metrics(self, cycles: List[Dict]) -> List[Dict]:
        """
        Compute detailed metrics for package cycles.

        Args:
            cycles: List of package cycles

        Returns:
            Cycles with added metrics
        """
        pkg_metrics = self.get_package_metrics()

        for cycle in cycles:
            packages = cycle['packages']

            # Aggregate metrics
            total_afferent = sum(pkg_metrics[p]['afferent_coupling'] for p in packages)
            total_efferent = sum(pkg_metrics[p]['efferent_coupling'] for p in packages)
            avg_instability = sum(pkg_metrics[p]['instability'] for p in packages) / len(packages)

            # External dependencies (dependencies to packages outside the cycle)
            cycle_pkg_set = set(packages)
            external_deps = 0

            for pkg_name in packages:
                pkg_idx = self.package_graph.vs.find(name=pkg_name).index
                for neighbor in self.package_graph.neighbors(pkg_idx, mode='out'):
                    neighbor_name = self.package_graph.vs[neighbor]['name']
                    if neighbor_name not in cycle_pkg_set:
                        external_deps += 1

            cycle['metrics'] = {
                'total_afferent': total_afferent,
                'total_efferent': total_efferent,
                'avg_instability': avg_instability,
                'external_dependencies': external_deps
            }

        return cycles

    def get_cycle_subgraph(self, cycle: Dict) -> ig.Graph:
        """
        Extract subgraph for a specific cycle.

        Args:
            cycle: Cycle dictionary with 'package_nodes'

        Returns:
            Subgraph containing only cycle packages
        """
        nodes = cycle['package_nodes']
        return self.package_graph.subgraph(nodes)

    def visualize_package_dependencies(self, output_file: str = None):
        """
        Create visualization of package dependencies.

        Args:
            output_file: Optional path to save visualization
        """
        if self.package_graph is None:
            self.build_package_graph()

        # Color packages by number of classes
        num_classes = [len(self.package_to_classes[pkg])
                      for pkg in self.package_graph.vs['name']]

        # Simplified visualization
        layout = self.package_graph.layout_fruchterman_reingold()

        visual_style = {
            'vertex_size': [min(20 + n * 2, 60) for n in num_classes],
            'vertex_label': self.package_graph.vs['name'],
            'vertex_label_size': 8,
            'edge_arrow_size': 0.5,
            'layout': layout,
            'bbox': (800, 800),
            'margin': 50
        }

        if output_file:
            ig.plot(self.package_graph, output_file, **visual_style)

        return visual_style
