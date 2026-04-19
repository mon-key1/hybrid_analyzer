"""Semantic similarity computation between code embeddings."""

from typing import List, Dict
import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Warning: sklearn not installed. Install with: pip install scikit-learn")
    cosine_similarity = None


class SimilarityCalculator:
    """Compute semantic similarity between classes using embeddings."""

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity in range [-1, 1] (typically [0, 1] for code)
        """
        if cosine_similarity is None:
            raise ImportError("sklearn required. Install with: pip install scikit-learn")

        # Ensure 2D shape
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)

        return float(cosine_similarity(emb1, emb2)[0, 0])

    @staticmethod
    def pairwise_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a set of embeddings.

        Args:
            embeddings: (N, D) array of N embeddings with dimension D

        Returns:
            (N, N) similarity matrix where element [i,j] is similarity between i and j
        """
        if cosine_similarity is None:
            raise ImportError("sklearn required. Install with: pip install scikit-learn")

        return cosine_similarity(embeddings)

    @staticmethod
    def average_pairwise_similarity(embeddings: np.ndarray, exclude_diagonal: bool = True) -> float:
        """
        Compute average pairwise similarity for a set of embeddings.

        Args:
            embeddings: (N, D) array of N embeddings
            exclude_diagonal: If True, exclude self-similarity (always 1.0)

        Returns:
            Average similarity score
        """
        if len(embeddings) < 2:
            return 1.0 if len(embeddings) == 1 else 0.0

        sim_matrix = SimilarityCalculator.pairwise_similarity_matrix(embeddings)

        if exclude_diagonal:
            # Get upper triangle excluding diagonal
            n = len(embeddings)
            triu_indices = np.triu_indices(n, k=1)
            similarities = sim_matrix[triu_indices]
        else:
            # Get all similarities
            similarities = sim_matrix.flatten()

        return float(np.mean(similarities))

    @staticmethod
    def compute_cycle_similarity(cycle_classes: List[str],
                                embeddings: Dict[str, np.ndarray]) -> float:
        """
        Compute average pairwise similarity for classes in a cycle.

        Args:
            cycle_classes: List of fully qualified class names in the cycle
            embeddings: Dictionary mapping class names to embeddings

        Returns:
            Average pairwise similarity (higher = more semantically coherent)
        """
        # Get embeddings for cycle classes
        cycle_embeddings = []
        for class_name in cycle_classes:
            if class_name in embeddings:
                cycle_embeddings.append(embeddings[class_name])
            else:
                print(f"Warning: No embedding found for {class_name}")

        if not cycle_embeddings:
            return 0.0

        if len(cycle_embeddings) == 1:
            return 1.0

        # Convert to array
        cycle_emb_array = np.array(cycle_embeddings)

        # Compute average pairwise similarity
        return SimilarityCalculator.average_pairwise_similarity(
            cycle_emb_array, exclude_diagonal=True
        )

    @staticmethod
    def find_most_similar_classes(target_class: str,
                                 embeddings: Dict[str, np.ndarray],
                                 top_k: int = 5) -> List[tuple[str, float]]:
        """
        Find the most similar classes to a target class.

        Args:
            target_class: Fully qualified name of target class
            embeddings: Dictionary of all class embeddings
            top_k: Number of top similar classes to return

        Returns:
            List of (class_name, similarity_score) tuples, sorted by similarity
        """
        if target_class not in embeddings:
            return []

        target_emb = embeddings[target_class]

        # Compute similarities to all other classes
        similarities = []
        for class_name, emb in embeddings.items():
            if class_name != target_class:
                sim = SimilarityCalculator.cosine_similarity(target_emb, emb)
                similarities.append((class_name, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    @staticmethod
    def compute_inter_package_similarity(package1_classes: List[str],
                                        package2_classes: List[str],
                                        embeddings: Dict[str, np.ndarray]) -> float:
        """
        Compute average similarity between two packages.

        Args:
            package1_classes: Classes in package 1
            package2_classes: Classes in package 2
            embeddings: Dictionary of embeddings

        Returns:
            Average cross-package similarity
        """
        similarities = []

        for class1 in package1_classes:
            if class1 not in embeddings:
                continue

            for class2 in package2_classes:
                if class2 not in embeddings:
                    continue

                sim = SimilarityCalculator.cosine_similarity(
                    embeddings[class1],
                    embeddings[class2]
                )
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0

    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Euclidean distance between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(emb1 - emb2))

    @staticmethod
    def manhattan_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Manhattan distance between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Manhattan distance
        """
        return float(np.sum(np.abs(emb1 - emb2)))
