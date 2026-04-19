"""RefactoringMiner wrapper for detecting refactorings in commits."""

import subprocess
import json
import os
from typing import List, Dict, Optional


class RefactoringDetector:
    """Wrapper for RefactoringMiner tool."""

    def __init__(self, jar_path: str = 'tools/RefactoringMiner-3.0.4.jar'):
        """
        Initialize RefactoringMiner wrapper.

        Args:
            jar_path: Path to RefactoringMiner JAR file
        """
        self.jar_path = jar_path

        if not os.path.exists(jar_path):
            print(f"Warning: RefactoringMiner JAR not found at {jar_path}")
            print("Download from: https://github.com/tsantalis/RefactoringMiner/releases")

    def is_available(self) -> bool:
        """Check if RefactoringMiner is available."""
        return os.path.exists(self.jar_path)

    def detect_refactorings(self,
                           repo_path: str,
                           commit_hash: str,
                           timeout: int = 60) -> List[Dict]:
        """
        Detect refactorings in a specific commit using RefactoringMiner.

        Args:
            repo_path: Path to git repository
            commit_hash: Commit hash to analyze
            timeout: Timeout in seconds

        Returns:
            List of refactoring dictionaries
        """
        if not self.is_available():
            return []

        try:
            # Run RefactoringMiner
            cmd = [
                'java', '-jar', self.jar_path,
                '-c', repo_path, commit_hash,
                '-json'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                print(f"RefactoringMiner error for commit {commit_hash}: {result.stderr}")
                return []

            # Parse JSON output
            output = result.stdout.strip()
            if not output:
                return []

            data = json.loads(output)

            # Extract refactorings
            refactorings = []
            for commit_data in data.get('commits', []):
                for ref in commit_data.get('refactorings', []):
                    refactorings.append(self._parse_refactoring(ref))

            return refactorings

        except subprocess.TimeoutExpired:
            print(f"RefactoringMiner timeout for commit {commit_hash}")
            return []
        except Exception as e:
            print(f"Error running RefactoringMiner: {e}")
            return []

    def _parse_refactoring(self, refactoring_data: Dict) -> Dict:
        """
        Parse refactoring data from RefactoringMiner output.

        Args:
            refactoring_data: Raw refactoring data from JSON

        Returns:
            Parsed refactoring dictionary
        """
        ref_type = refactoring_data.get('type', 'Unknown')
        description = refactoring_data.get('description', '')

        # Extract source and target locations
        left_locations = refactoring_data.get('leftSideLocations', [])
        right_locations = refactoring_data.get('rightSideLocations', [])

        source_class = None
        target_class = None

        if left_locations:
            source_class = left_locations[0].get('filePath')

        if right_locations:
            target_class = right_locations[0].get('filePath')

        return {
            'type': ref_type,
            'description': description,
            'source_class': source_class,
            'target_class': target_class,
            'is_structural': self._is_structural_refactoring(ref_type)
        }

    def _is_structural_refactoring(self, ref_type: str) -> bool:
        """
        Check if refactoring type affects structure (relevant for cycles).

        Args:
            ref_type: Refactoring type

        Returns:
            True if structural refactoring
        """
        structural_types = [
            'Move Class',
            'Move Method',
            'Move Field',
            'Extract Interface',
            'Extract Superclass',
            'Extract Class',
            'Inline Class',
            'Move And Rename Class',
            'Change Package',
            'Extract Subclass'
        ]

        return ref_type in structural_types

    def detect_refactorings_in_range(self,
                                     repo_path: str,
                                     since: Optional[str] = None,
                                     until: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Detect refactorings in all commits in a date range.

        Args:
            repo_path: Path to git repository
            since: Start date (ISO format)
            until: End date (ISO format)

        Returns:
            Dictionary mapping commit hash to list of refactorings
        """
        if not self.is_available():
            return {}

        # First, get commits in range using pydriller
        from .git_analyzer import GitHistoryAnalyzer

        git_analyzer = GitHistoryAnalyzer(repo_path)
        commits = git_analyzer.extract_commits_in_range(since, until, only_refactorings=True)

        # Detect refactorings for each commit
        refactorings_by_commit = {}

        for commit in commits:
            commit_hash = commit['hash']
            refactorings = self.detect_refactorings(repo_path, commit_hash)

            if refactorings:
                refactorings_by_commit[commit_hash] = refactorings

        return refactorings_by_commit

    def filter_cycle_related_refactorings(self,
                                         refactorings: List[Dict],
                                         cycle_classes: List[str]) -> List[Dict]:
        """
        Filter refactorings that affect classes in a cycle.

        Args:
            refactorings: List of refactoring dictionaries
            cycle_classes: List of fully qualified class names in cycle

        Returns:
            Filtered list of refactorings
        """
        # Convert cycle classes to file paths (approximate)
        cycle_files = set()
        for class_name in cycle_classes:
            # Convert package.ClassName to path/ClassName.java
            file_path = class_name.replace('.', '/') + '.java'
            cycle_files.add(file_path)

        # Filter refactorings
        related = []
        for ref in refactorings:
            source = ref.get('source_class', '')
            target = ref.get('target_class', '')

            # Check if source or target matches any cycle file
            if any(cf in source or cf in target for cf in cycle_files):
                related.append(ref)

        return related
