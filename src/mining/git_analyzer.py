"""Git history analysis for cycle elimination patterns."""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re

try:
    from pydriller import Repository
except ImportError:
    print("Warning: pydriller not installed. Install with: pip install pydriller")
    Repository = None


class GitHistoryAnalyzer:
    """Analyze git history for cycle elimination patterns."""

    def __init__(self, repo_path: str):
        """
        Initialize git history analyzer.

        Args:
            repo_path: Path to git repository
        """
        if Repository is None:
            raise ImportError("pydriller required. Install with: pip install pydriller")

        self.repo_path = repo_path
        self.repo = Repository(repo_path)

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse date string to datetime object.

        Args:
            date_str: ISO date string (e.g., '2021-01-01') or None

        Returns:
            datetime object or None
        """
        if date_str is None:
            return None

        try:
            # Parse ISO date string
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            # Try with time included
            try:
                return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # If parsing fails, return None
                print(f"Warning: Could not parse date string: {date_str}")
                return None

    def extract_commits_in_range(self,
                                 since: Optional[str] = None,
                                 until: Optional[str] = None,
                                 only_refactorings: bool = False) -> List[Dict]:
        """
        Extract commit metadata in a date range.

        Args:
            since: ISO date string (e.g., '2021-01-01')
            until: ISO date string
            only_refactorings: If True, filter to likely refactoring commits

        Returns:
            List of commit dictionaries
        """
        commits = []

        # Convert date strings to datetime objects
        since_dt = self._parse_date(since)
        until_dt = self._parse_date(until)

        for commit in self.repo.traverse_commits(since=since_dt, to=until_dt):
            commit_data = {
                'hash': commit.hash,
                'date': commit.committer_date,
                'message': commit.msg,
                'author': commit.author.name,
                'files_changed': [m.new_path for m in commit.modified_files
                                 if m.new_path and m.new_path.endswith('.java')],
                'num_files': len([m for m in commit.modified_files
                                 if m.new_path and m.new_path.endswith('.java')]),
                'insertions': commit.insertions,
                'deletions': commit.deletions
            }

            # Analyze commit message
            message_analysis = self.analyze_commit_message(commit.msg)
            commit_data.update(message_analysis)

            # Filter if needed
            if only_refactorings and not message_analysis['is_refactoring']:
                continue

            commits.append(commit_data)

        return commits

    def analyze_commit_message(self, message: str) -> Dict[str, any]:
        """
        Analyze commit message for refactoring keywords.

        Args:
            message: Commit message

        Returns:
            Dictionary with:
            - is_refactoring: bool
            - mentions_cycle: bool
            - keywords_found: List[str]
        """
        refactoring_keywords = [
            'refactor', 'restructure', 'decouple', 'break cycle',
            'remove circular', 'circular dependency', 'dependency cycle',
            'extract interface', 'move class', 'modularize',
            'split', 'merge', 'rename', 'extract', 'inline',
            'move method', 'pull up', 'push down'
        ]

        cycle_keywords = [
            'cycle', 'circular', 'cyclic', 'loop'
        ]

        message_lower = message.lower()

        # Find keywords
        found_refactoring = [kw for kw in refactoring_keywords if kw in message_lower]
        found_cycle = [kw for kw in cycle_keywords if kw in message_lower]

        return {
            'is_refactoring': len(found_refactoring) > 0,
            'mentions_cycle': len(found_cycle) > 0,
            'keywords_found': found_refactoring + found_cycle
        }

    def get_file_commit_frequency(self,
                                  since: Optional[str] = None,
                                  until: Optional[str] = None) -> Dict[str, int]:
        """
        Get commit frequency for each file.

        Args:
            since: Start date
            until: End date

        Returns:
            Dictionary mapping file path to commit count
        """
        file_frequency = {}

        # Convert date strings to datetime objects
        since_dt = self._parse_date(since)
        until_dt = self._parse_date(until)

        for commit in self.repo.traverse_commits(since=since_dt, to=until_dt):
            for modified_file in commit.modified_files:
                if modified_file.new_path and modified_file.new_path.endswith('.java'):
                    file_path = modified_file.new_path
                    file_frequency[file_path] = file_frequency.get(file_path, 0) + 1

        return file_frequency

    def get_file_authors(self,
                        since: Optional[str] = None,
                        until: Optional[str] = None) -> Dict[str, set]:
        """
        Get unique authors for each file.

        Args:
            since: Start date
            until: End date

        Returns:
            Dictionary mapping file path to set of author names
        """
        file_authors = {}

        # Convert date strings to datetime objects
        since_dt = self._parse_date(since)
        until_dt = self._parse_date(until)

        for commit in self.repo.traverse_commits(since=since_dt, to=until_dt):
            for modified_file in commit.modified_files:
                if modified_file.new_path and modified_file.new_path.endswith('.java'):
                    file_path = modified_file.new_path
                    if file_path not in file_authors:
                        file_authors[file_path] = set()
                    file_authors[file_path].add(commit.author.name)

        return file_authors

    def compute_file_churn(self,
                          since: Optional[str] = None,
                          until: Optional[str] = None) -> Dict[str, Dict]:
        """
        Compute code churn (additions + deletions) for each file.

        Args:
            since: Start date
            until: End date

        Returns:
            Dictionary mapping file path to churn metrics
        """
        file_churn = {}

        # Convert date strings to datetime objects
        since_dt = self._parse_date(since)
        until_dt = self._parse_date(until)

        for commit in self.repo.traverse_commits(since=since_dt, to=until_dt):
            for modified_file in commit.modified_files:
                if modified_file.new_path and modified_file.new_path.endswith('.java'):
                    file_path = modified_file.new_path

                    if file_path not in file_churn:
                        file_churn[file_path] = {
                            'additions': 0,
                            'deletions': 0,
                            'total_churn': 0,
                            'commits': 0
                        }

                    file_churn[file_path]['additions'] += modified_file.added_lines
                    file_churn[file_path]['deletions'] += modified_file.deleted_lines
                    file_churn[file_path]['total_churn'] += (modified_file.added_lines +
                                                             modified_file.deleted_lines)
                    file_churn[file_path]['commits'] += 1

        return file_churn

    def find_refactoring_commits(self,
                                since: Optional[str] = None,
                                until: Optional[str] = None) -> List[Dict]:
        """
        Find commits likely to be refactorings.

        Heuristics:
        - Message contains refactoring keywords
        - Moderate number of files changed (2-20)
        - Balance of additions and deletions

        Args:
            since: Start date
            until: End date

        Returns:
            List of refactoring commit dictionaries
        """
        refactoring_commits = []

        # Convert date strings to datetime objects
        since_dt = self._parse_date(since)
        until_dt = self._parse_date(until)

        for commit in self.repo.traverse_commits(since=since_dt, to=until_dt):
            message_analysis = self.analyze_commit_message(commit.msg)

            # Check heuristics
            num_java_files = len([m for m in commit.modified_files
                                 if m.new_path and m.new_path.endswith('.java')])

            # Heuristic: refactorings usually change multiple files but not too many
            is_moderate_size = 2 <= num_java_files <= 20

            # Heuristic: refactorings have balanced additions/deletions
            total_changes = commit.insertions + commit.deletions
            if total_changes > 0:
                balance_ratio = min(commit.insertions, commit.deletions) / total_changes
                is_balanced = balance_ratio > 0.3
            else:
                is_balanced = False

            # Classify as refactoring if meets criteria
            if message_analysis['is_refactoring'] or (is_moderate_size and is_balanced):
                refactoring_commits.append({
                    'hash': commit.hash,
                    'date': commit.committer_date,
                    'message': commit.msg,
                    'author': commit.author.name,
                    'num_files': num_java_files,
                    'insertions': commit.insertions,
                    'deletions': commit.deletions,
                    'keywords': message_analysis['keywords_found'],
                    'mentions_cycle': message_analysis['mentions_cycle']
                })

        return refactoring_commits

    def get_repository_stats(self) -> Dict:
        """
        Get basic repository statistics.

        Returns:
            Dictionary with repository stats
        """
        commit_count = 0
        authors = set()
        first_commit_date = None
        last_commit_date = None

        for commit in self.repo.traverse_commits():
            commit_count += 1
            authors.add(commit.author.name)

            if first_commit_date is None or commit.committer_date < first_commit_date:
                first_commit_date = commit.committer_date

            if last_commit_date is None or commit.committer_date > last_commit_date:
                last_commit_date = commit.committer_date

        return {
            'total_commits': commit_count,
            'num_authors': len(authors),
            'first_commit': first_commit_date,
            'last_commit': last_commit_date
        }
