"""Code embedding generation using pre-trained models."""

import os
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
    SentenceTransformer = None

from ..parser.java_parser import ParsedClass


class CodeEmbedder:
    """Generate semantic embeddings for code using pre-trained models."""

    def __init__(self, model_name: str = 'microsoft/unixcoder-base', use_onnx: bool = False):
        """
        Initialize embedding model.

        Args:
            model_name: Model name from HuggingFace
                       - 'microsoft/unixcoder-base' (768-dim)
                       - 'Salesforce/codet5p-110m-embedding' (256-dim)
            use_onnx: Convert to ONNX for speedup (not implemented yet)
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")

        self.model_name = model_name
        self.use_onnx = use_onnx

        print(f"Loading embedding model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            print(f"✓ Loaded {model_name}")
        except Exception as e:
            print(f"Warning: Could not load {model_name}")
            print(f"Falling back to 'all-MiniLM-L6-v2' model...")
            # Fallback to a simpler model
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"✓ Loaded fallback model 'all-MiniLM-L6-v2'")
            except Exception as e2:
                raise RuntimeError(f"Could not load any embedding model. Error: {e2}")

        if use_onnx:
            print("Warning: ONNX optimization not yet implemented")

    def create_semantic_text(self, parsed_class: ParsedClass) -> str:
        """
        Create text representation for embedding.

        Format includes package, class name, methods, fields, javadoc, and imports.
        Kept under 512 tokens for model limits.

        Args:
            parsed_class: ParsedClass object

        Returns:
            String representation for embedding
        """
        parts = []

        # Package
        if parsed_class.package_name:
            parts.append(f"[Package: {parsed_class.package_name}]")

        # Class name
        parts.append(f"[Class: {parsed_class.class_name}]")

        # Class type
        if parsed_class.is_interface:
            parts.append("[Type: Interface]")
        elif parsed_class.is_abstract:
            parts.append("[Type: Abstract Class]")

        # Methods (limit to first 10)
        if parsed_class.methods:
            method_names = [m.name for m in parsed_class.methods[:10]]
            parts.append(f"[Methods: {', '.join(method_names)}]")

        # Fields (limit to first 10)
        if parsed_class.fields:
            field_names = [f.name for f in parsed_class.fields[:10]]
            parts.append(f"[Fields: {', '.join(field_names)}]")

        # JavaDoc (first 2 lines if available)
        if parsed_class.javadoc:
            javadoc_lines = parsed_class.javadoc.strip().split('\n')[:2]
            javadoc_text = ' '.join(line.strip() for line in javadoc_lines)
            if javadoc_text:
                parts.append(f"[JavaDoc: {javadoc_text[:200]}]")

        # Key imports (filter to non-standard library, limit to 5)
        if parsed_class.imports:
            key_imports = [imp for imp in parsed_class.imports
                          if not imp.startswith('java.lang.')
                          and not imp.startswith('java.util.')
                          and not imp.startswith('javax.')][:5]
            if key_imports:
                parts.append(f"[Imports: {', '.join(key_imports)}]")

        # Combine all parts
        semantic_text = ' '.join(parts)

        # Truncate if too long (approximately 512 tokens ~= 2048 chars)
        if len(semantic_text) > 2048:
            semantic_text = semantic_text[:2048]

        return semantic_text

    def embed_class(self, semantic_text: str) -> np.ndarray:
        """
        Generate embedding vector for a single class.

        Args:
            semantic_text: Text representation of the class

        Returns:
            Embedding vector (768-dim for UniXcoder, 256-dim for CodeT5+)
        """
        return self.model.encode(semantic_text, convert_to_numpy=True)

    def embed_batch(self, semantic_texts: List[str], batch_size: int = 16,
                   show_progress: bool = True) -> np.ndarray:
        """
        Batch embedding for efficiency.

        Args:
            semantic_texts: List of text representations
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Array of embeddings (N, D) where N is number of texts
        """
        return self.model.encode(
            semantic_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

    def embed_parsed_classes(self, parsed_classes: List[ParsedClass],
                            batch_size: int = 16,
                            cache_dir: Optional[str] = None) -> dict[str, np.ndarray]:
        """
        Embed multiple parsed classes with optional caching.

        Args:
            parsed_classes: List of ParsedClass objects
            batch_size: Batch size for encoding
            cache_dir: Directory to cache embeddings (optional)

        Returns:
            Dictionary mapping fully_qualified_name to embedding vector
        """
        embeddings = {}

        # Prepare for caching
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        texts_to_embed = []
        classes_to_embed = []

        # Check cache
        for parsed_class in parsed_classes:
            cache_key = parsed_class.fully_qualified_name

            if cache_dir:
                cache_file = os.path.join(cache_dir, f"{cache_key.replace('.', '_')}.npy")

                if os.path.exists(cache_file):
                    # Load from cache
                    try:
                        embeddings[cache_key] = np.load(cache_file)
                        continue
                    except Exception as e:
                        print(f"Warning: Could not load cache for {cache_key}: {e}")

            # Need to compute embedding
            semantic_text = self.create_semantic_text(parsed_class)
            texts_to_embed.append(semantic_text)
            classes_to_embed.append(parsed_class)

        # Batch embed missing embeddings
        if texts_to_embed:
            print(f"Generating {len(texts_to_embed)} embeddings...")
            new_embeddings = self.embed_batch(texts_to_embed, batch_size=batch_size)

            # Store and cache
            for parsed_class, embedding in zip(classes_to_embed, new_embeddings):
                cache_key = parsed_class.fully_qualified_name
                embeddings[cache_key] = embedding

                if cache_dir:
                    cache_file = os.path.join(cache_dir, f"{cache_key.replace('.', '_')}.npy")
                    np.save(cache_file, embedding)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        # Test with a simple string
        test_emb = self.embed_class("test")
        return len(test_emb)