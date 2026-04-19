"""Java source code parser using tree-sitter and javalang."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
import os
from pathlib import Path

try:
    from tree_sitter import Language, Parser
    import tree_sitter_java as tsjava
except ImportError:
    print("Warning: tree-sitter not available. Install with: pip install tree-sitter tree-sitter-java")
    Parser = None

try:
    import javalang
except ImportError:
    print("Warning: javalang not available. Install with: pip install javalang")
    javalang = None


@dataclass
class MethodInfo:
    """Information about a Java method."""
    name: str
    parameters: List[str] = field(default_factory=list)
    return_type: str = ""
    calls: List[str] = field(default_factory=list)
    javadoc: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)


@dataclass
class FieldInfo:
    """Information about a Java field."""
    name: str
    type: str
    javadoc: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)


@dataclass
class ParsedClass:
    """Parsed Java class information."""
    class_name: str
    package_name: str
    fully_qualified_name: str
    imports: List[str] = field(default_factory=list)
    methods: List[MethodInfo] = field(default_factory=list)
    fields: List[FieldInfo] = field(default_factory=list)
    javadoc: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    superclass: Optional[str] = None
    interfaces: List[str] = field(default_factory=list)
    is_interface: bool = False
    is_abstract: bool = False
    file_path: str = ""


class JavaParser:
    """Parse Java source code using tree-sitter and javalang."""

    def __init__(self):
        """Initialize the parser with tree-sitter Java grammar."""
        if Parser is None:
            raise ImportError("tree-sitter not installed")

        self.parser = Parser()
        try:
            # Use tree-sitter-java language
            JAVA_LANGUAGE = Language(tsjava.language(), name="java")
            self.parser.set_language(JAVA_LANGUAGE)
        except Exception as e:
            print(f"Warning: Could not initialize tree-sitter-java: {e}")
            self.parser = None

    def parse_file(self, file_path: str) -> Optional[ParsedClass]:
        """
        Parse a single Java file.

        Args:
            file_path: Path to the Java source file

        Returns:
            ParsedClass object or None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()

            # Use javalang as primary parser (more robust for Java-specific constructs)
            return self._parse_with_javalang(source_code, file_path)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def _parse_with_javalang(self, source_code: str, file_path: str) -> Optional[ParsedClass]:
        """Parse using javalang library."""
        if javalang is None:
            return None

        try:
            tree = javalang.parse.parse(source_code)
        except Exception as e:
            # Try to extract basic info even if full parsing fails
            return self._fallback_parse(source_code, file_path)

        package_name = tree.package.name if tree.package else ""
        # Preserve wildcard information in imports
        imports = []
        if tree.imports:
            for imp in tree.imports:
                if imp.wildcard:
                    imports.append(f"{imp.path}.*")
                else:
                    imports.append(imp.path)

        # Find the main class/interface
        classes = []
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            classes.append(('class', node))
        for path, node in tree.filter(javalang.tree.InterfaceDeclaration):
            classes.append(('interface', node))

        if not classes:
            return None

        # Take the first class/interface (main one)
        class_type, class_node = classes[0]

        class_name = class_node.name
        fully_qualified_name = f"{package_name}.{class_name}" if package_name else class_name

        # Extract methods
        methods = []
        for method in class_node.methods or []:
            method_info = MethodInfo(
                name=method.name,
                parameters=[param.type.name if hasattr(param.type, 'name') else str(param.type)
                           for param in (method.parameters or [])],
                return_type=method.return_type.name if method.return_type and hasattr(method.return_type, 'name') else "",
                modifiers=method.modifiers or []
            )

            # Extract method calls (simplified)
            if method.body:
                method_info.calls = self._extract_method_calls(method.body)

            methods.append(method_info)

        # Extract fields
        fields = []
        for field_decl in class_node.fields or []:
            for declarator in field_decl.declarators:
                field_info = FieldInfo(
                    name=declarator.name,
                    type=field_decl.type.name if hasattr(field_decl.type, 'name') else str(field_decl.type),
                    modifiers=field_decl.modifiers or []
                )
                fields.append(field_info)

        # Extract dependencies
        dependencies = self._extract_dependencies(tree, imports, package_name)

        # Get superclass and interfaces
        superclass = None
        interfaces = []

        if hasattr(class_node, 'extends') and class_node.extends:
            # For interfaces, extends is a list (interface can extend multiple interfaces)
            # For classes, extends is a single ReferenceType (class can extend only one class)
            if class_type == 'interface':
                # Interface - extends is a list
                for ext in class_node.extends:
                    simple_name = ext.name if hasattr(ext, 'name') else str(ext)
                    resolved = self._resolve_type(simple_name, imports, package_name)
                    if resolved:
                        interfaces.append(resolved)
            else:
                # Class - extends is a single superclass
                simple_name = class_node.extends.name if hasattr(class_node.extends, 'name') else str(class_node.extends)
                superclass = self._resolve_type(simple_name, imports, package_name)

        if hasattr(class_node, 'implements') and class_node.implements:
            for impl in class_node.implements:
                simple_name = impl.name if hasattr(impl, 'name') else str(impl)
                # Resolve to fully qualified name
                resolved = self._resolve_type(simple_name, imports, package_name)
                if resolved:
                    interfaces.append(resolved)

        # Extract JavaDoc
        javadoc = class_node.documentation if hasattr(class_node, 'documentation') else None

        return ParsedClass(
            class_name=class_name,
            package_name=package_name,
            fully_qualified_name=fully_qualified_name,
            imports=imports,
            methods=methods,
            fields=fields,
            javadoc=javadoc,
            dependencies=dependencies,
            superclass=superclass,
            interfaces=interfaces,
            is_interface=(class_type == 'interface'),
            is_abstract='abstract' in (class_node.modifiers or []),
            file_path=file_path
        )

    def _extract_method_calls(self, body) -> List[str]:
        """Extract method calls from method body (simplified)."""
        calls = []
        try:
            for path, node in body.filter(javalang.tree.MethodInvocation):
                if node.member:
                    calls.append(node.member)
        except:
            pass
        return calls

    def _extract_dependencies(self, tree, imports: List[str], package_name: str) -> List[str]:
        """Extract class dependencies from the AST."""
        dependencies = set()

        # Add imports (these are already fully qualified or wildcard)
        for imp in imports:
            dependencies.add(imp)

        # Extract type references - store simple names
        # SymbolResolver will resolve them correctly using class_index
        try:
            for path, node in tree.filter(javalang.tree.ReferenceType):
                if hasattr(node, 'name') and node.name:
                    # Store simple name - will be resolved by SymbolResolver
                    dependencies.add(node.name)
        except:
            pass

        return list(dependencies)

    def _resolve_type(self, type_name: str, imports: List[str], package_name: str) -> Optional[str]:
        """Resolve a type name to its fully qualified name."""
        # Check if already fully qualified
        if '.' in type_name:
            return type_name

        # Check imports for exact matches first
        for imp in imports:
            if imp.endswith('.' + type_name):
                # Exact match import (e.g., import java.util.List)
                return imp

        # Try same package first (higher priority than wildcard)
        if package_name:
            same_package_candidate = f"{package_name}.{type_name}"
            # Return same package guess - this is likely correct for inner references
            return same_package_candidate

        # If no package, check wildcard imports
        # NOTE: This may resolve incorrectly when multiple wildcards exist.
        # SymbolResolver will verify against class_index later.
        for imp in imports:
            if imp.endswith('.*'):
                # Wildcard import - return first match
                # SymbolResolver will filter out incorrect guesses
                return imp[:-2] + '.' + type_name

        return type_name

    def _fallback_parse(self, source_code: str, file_path: str) -> Optional[ParsedClass]:
        """Fallback parsing using simple regex when javalang fails."""
        import re

        # Extract package
        package_match = re.search(r'package\s+([\w.]+)\s*;', source_code)
        package_name = package_match.group(1) if package_match else ""

        # Extract class name
        class_match = re.search(r'(?:public\s+)?(?:abstract\s+)?(?:class|interface)\s+(\w+)', source_code)
        if not class_match:
            return None

        class_name = class_match.group(1)
        fully_qualified_name = f"{package_name}.{class_name}" if package_name else class_name

        # Extract imports
        imports = re.findall(r'import\s+([\w.]+)\s*;', source_code)

        return ParsedClass(
            class_name=class_name,
            package_name=package_name,
            fully_qualified_name=fully_qualified_name,
            imports=imports,
            file_path=file_path
        )

    def extract_identifiers(self, parsed_class: ParsedClass) -> List[str]:
        """
        Extract all identifiers for semantic analysis.

        Returns:
            List of identifiers: [package, class, methods, fields]
        """
        identifiers = []

        if parsed_class.package_name:
            identifiers.append(parsed_class.package_name)

        identifiers.append(parsed_class.class_name)

        for method in parsed_class.methods:
            identifiers.append(method.name)

        for field in parsed_class.fields:
            identifiers.append(field.name)

        return identifiers

    def extract_dependencies(self, parsed_class: ParsedClass) -> List[Tuple[str, str, str]]:
        """
        Extract dependency edges from parsed class.

        Returns:
            List of (source_class, target_class, edge_type) tuples
        """
        edges = []
        source = parsed_class.fully_qualified_name

        # Inheritance
        if parsed_class.superclass:
            edges.append((source, parsed_class.superclass, 'extends'))

        # Interface implementation
        for interface in parsed_class.interfaces:
            edges.append((source, interface, 'implements'))

        # Field types
        for field in parsed_class.fields:
            # Resolve field type
            target = self._resolve_type(field.type, parsed_class.imports, parsed_class.package_name)
            if target and target != source:
                edges.append((source, target, 'field_type'))

        # General dependencies
        for dep in parsed_class.dependencies:
            if dep != source and not any(e[1] == dep for e in edges):
                edges.append((source, dep, 'uses'))

        return edges