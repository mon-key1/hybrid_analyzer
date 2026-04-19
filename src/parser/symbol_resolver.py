"""Symbol resolution for Java types and dependencies."""

from typing import Dict, List, Optional, Set
from .java_parser import ParsedClass


class SymbolResolver:
    """Resolve Java type references to fully qualified names."""

    def __init__(self):
        """Initialize the symbol resolver."""
        self.class_index: Dict[str, ParsedClass] = {}
        self.package_index: Dict[str, List[str]] = {}

    def index_classes(self, parsed_classes: List[ParsedClass]):
        """
        Build an index of all parsed classes.

        Args:
            parsed_classes: List of ParsedClass objects
        """
        self.class_index = {}
        self.package_index = {}

        for parsed_class in parsed_classes:
            # Index by fully qualified name
            self.class_index[parsed_class.fully_qualified_name] = parsed_class

            # Index by simple name
            if parsed_class.class_name not in self.class_index:
                self.class_index[parsed_class.class_name] = parsed_class

            # Index by package
            if parsed_class.package_name not in self.package_index:
                self.package_index[parsed_class.package_name] = []
            self.package_index[parsed_class.package_name].append(
                parsed_class.fully_qualified_name
            )

    def resolve_type(self, type_name: str, context_class: ParsedClass) -> Optional[str]:
        """
        Resolve a type name to its fully qualified name.

        Args:
            type_name: The type name to resolve
            context_class: The class context for resolution

        Returns:
            Fully qualified type name or None if cannot be resolved
        """
        # Already fully qualified
        if type_name in self.class_index:
            return type_name

        # Check imports
        for imp in context_class.imports:
            if imp.endswith('.' + type_name):
                return imp
            elif imp.endswith('.*'):
                # Wildcard import
                package = imp[:-2]
                candidate = f"{package}.{type_name}"
                if candidate in self.class_index:
                    return candidate

        # Same package
        if context_class.package_name:
            candidate = f"{context_class.package_name}.{type_name}"
            if candidate in self.class_index:
                return candidate

        # Java built-in types
        if type_name in ['String', 'Integer', 'Long', 'Double', 'Float',
                        'Boolean', 'Character', 'Byte', 'Short', 'Object',
                        'List', 'Set', 'Map', 'ArrayList', 'HashMap', 'HashSet']:
            return f"java.lang.{type_name}" if type_name in ['String', 'Integer', 'Object'] else f"java.util.{type_name}"

        # Cannot resolve - return as is
        return type_name

    def get_class_dependencies(self, parsed_class: ParsedClass) -> Set[str]:
        """
        Get all dependencies for a class.

        Args:
            parsed_class: The class to analyze

        Returns:
            Set of fully qualified class names that this class depends on
        """
        dependencies = set()

        # Superclass
        if parsed_class.superclass:
            resolved = self.resolve_type(parsed_class.superclass, parsed_class)
            if resolved:
                dependencies.add(resolved)

        # Interfaces
        for interface in parsed_class.interfaces:
            resolved = self.resolve_type(interface, parsed_class)
            if resolved:
                dependencies.add(resolved)

        # Field types
        for field in parsed_class.fields:
            resolved = self.resolve_type(field.type, parsed_class)
            if resolved and resolved != parsed_class.fully_qualified_name:
                dependencies.add(resolved)

        # Method parameter and return types
        for method in parsed_class.methods:
            if method.return_type:
                resolved = self.resolve_type(method.return_type, parsed_class)
                if resolved and resolved != parsed_class.fully_qualified_name:
                    dependencies.add(resolved)

            for param_type in method.parameters:
                resolved = self.resolve_type(param_type, parsed_class)
                if resolved and resolved != parsed_class.fully_qualified_name:
                    dependencies.add(resolved)

        # Add dependencies extracted by parser (imports, type references, etc.)
        for dep in parsed_class.dependencies:
            # Skip wildcard imports - they are not actual dependencies
            if dep.endswith('.*'):
                continue

            # If already fully qualified (exact import), add directly
            if '.' in dep and not dep.endswith('.*'):
                if dep != parsed_class.fully_qualified_name:
                    dependencies.add(dep)
            else:
                # Simple name - try to resolve
                resolved = self.resolve_type(dep, parsed_class)
                if resolved and resolved != parsed_class.fully_qualified_name:
                    dependencies.add(resolved)

        # Filter out java.lang and primitives
        dependencies = {dep for dep in dependencies
                       if not dep.startswith('java.lang.')
                       and not dep.startswith('java.util.')
                       and dep in self.class_index}

        return dependencies

    def resolve_all_dependencies(self, parsed_classes: List[ParsedClass]) -> Dict[str, Set[str]]:
        """
        Resolve dependencies for all classes.

        Args:
            parsed_classes: List of all parsed classes

        Returns:
            Dictionary mapping class name to set of dependencies
        """
        self.index_classes(parsed_classes)

        dependency_map = {}
        for parsed_class in parsed_classes:
            deps = self.get_class_dependencies(parsed_class)
            dependency_map[parsed_class.fully_qualified_name] = deps

        return dependency_map