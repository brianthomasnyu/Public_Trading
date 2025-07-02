"""
Repository Management Agent - Autonomous Codebase Management
==========================================================

This agent manages repository operations, code updates, and version control
through intelligent automation and change tracking.

CRITICAL SYSTEM POLICY: NO TRADING DECISIONS OR RECOMMENDATIONS
This agent only performs repository management, code updates, and version
control operations. No trading advice is provided.
"""

import asyncio
import logging
import json
import os
import subprocess
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import git
from pathlib import Path
import difflib
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeChange:
    """Represents a code change or update"""
    file_path: str
    change_type: str  # 'add', 'modify', 'delete', 'refactor'
    description: str
    priority: int = 1
    complexity: str = 'low'  # 'low', 'medium', 'high'
    estimated_effort: int = 1  # hours
    dependencies: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class RepositoryStatus:
    """Repository status and health information"""
    branch: str
    last_commit: str
    last_commit_date: datetime
    uncommitted_changes: int
    untracked_files: int
    ahead_behind: Tuple[int, int]
    health_score: float

class RepositoryManagementAgent:
    """
    Autonomous repository management agent
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "RepositoryManagementAgent"
        self.version = "1.0.0"
        self.repo_path = config.get("repo_path", ".")
        self.repo = None
        self.pending_changes = []
        self.change_history = []
        
    async def initialize(self):
        """Initialize the repository management agent"""
        logger.info(f"Initializing {self.name} v{self.version}")
        
        # AI REASONING: Initialize repository management capabilities
        # PSEUDOCODE:
        # 1. Validate repository path and git configuration
        # 2. Initialize git repository object and status tracking
        # 3. Set up change detection and monitoring systems
        # 4. Configure automated backup and rollback mechanisms
        # 5. Initialize code quality and testing frameworks
        # 6. Set up dependency management and security scanning
        # 7. Configure branch protection and merge policies
        # 8. Initialize MCP communication for repository updates
        
        try:
            self.repo = git.Repo(self.repo_path)
            logger.info(f"Repository initialized: {self.repo_path}")
            
            # Validate repository health
            await self._validate_repository_health()
            
        except git.InvalidGitRepositoryError:
            logger.error(f"Invalid git repository: {self.repo_path}")
            raise
        except Exception as e:
            logger.error(f"Error initializing repository: {e}")
            raise
            
        logger.info(f"{self.name} initialized successfully")
    
    async def _validate_repository_health(self):
        """
        Validate repository health and configuration
        """
        # AI REASONING: Repository health validation
        # PSEUDOCODE:
        # 1. Check git repository integrity and consistency
        # 2. Validate remote repository connectivity
        # 3. Assess branch structure and protection rules
        # 4. Check for merge conflicts and unresolved issues
        # 5. Validate commit history and log integrity
        # 6. Assess repository size and performance metrics
        # 7. Check for security vulnerabilities in dependencies
        # 8. Generate health score and recommendations
        
        try:
            # Check if repository is valid
            if not self.repo.git_dir:
                raise ValueError("Invalid git repository")
            
            # Check remote connectivity
            if self.repo.remotes:
                for remote in self.repo.remotes:
                    try:
                        self.repo.git.fetch(remote.name)
                        logger.info(f"Remote {remote.name} is accessible")
                    except Exception as e:
                        logger.warning(f"Remote {remote.name} connectivity issue: {e}")
            
            # Check for uncommitted changes
            uncommitted = len(self.repo.index.diff(None)) + len(self.repo.untracked_files)
            if uncommitted > 0:
                logger.info(f"Found {uncommitted} uncommitted changes")
                
        except Exception as e:
            logger.error(f"Repository health validation failed: {e}")
            raise
    
    async def get_repository_status(self) -> RepositoryStatus:
        """
        Get current repository status and health
        """
        # AI REASONING: Repository status analysis
        # PSEUDOCODE:
        # 1. Analyze current branch and commit information
        # 2. Calculate uncommitted changes and untracked files
        # 3. Assess ahead/behind status with remote
        # 4. Evaluate repository health metrics
        # 5. Check for potential issues and conflicts
        # 6. Generate status summary and recommendations
        # 7. Update repository monitoring data
        
        try:
            # Get current branch
            current_branch = self.repo.active_branch.name
            
            # Get last commit info
            last_commit = self.repo.head.commit
            last_commit_hash = last_commit.hexsha[:8]
            last_commit_date = datetime.fromtimestamp(last_commit.committed_date)
            
            # Count uncommitted changes
            uncommitted_changes = len(self.repo.index.diff(None))
            untracked_files = len(self.repo.untracked_files)
            
            # Get ahead/behind status
            ahead_behind = (0, 0)
            if self.repo.remotes:
                try:
                    ahead_behind = self.repo.iter_commits(f"{current_branch}..origin/{current_branch}")
                    behind = len(list(ahead_behind))
                    ahead_behind = self.repo.iter_commits(f"origin/{current_branch}..{current_branch}")
                    ahead = len(list(ahead_behind))
                    ahead_behind = (ahead, behind)
                except:
                    ahead_behind = (0, 0)
            
            # Calculate health score
            health_score = self._calculate_health_score(
                uncommitted_changes, untracked_files, ahead_behind
            )
            
            status = RepositoryStatus(
                branch=current_branch,
                last_commit=last_commit_hash,
                last_commit_date=last_commit_date,
                uncommitted_changes=uncommitted_changes,
                untracked_files=untracked_files,
                ahead_behind=ahead_behind,
                health_score=health_score
            )
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting repository status: {e}")
            raise
    
    def _calculate_health_score(self, uncommitted: int, untracked: int, ahead_behind: Tuple[int, int]) -> float:
        """
        Calculate repository health score
        """
        # AI REASONING: Health score calculation
        # PSEUDOCODE:
        # 1. Assess uncommitted changes impact on health
        # 2. Evaluate untracked files and their significance
        # 3. Consider ahead/behind status with remote
        # 4. Factor in repository age and activity
        # 5. Apply weighting based on change criticality
        # 6. Generate normalized health score (0-1)
        
        base_score = 1.0
        
        # Penalize uncommitted changes
        if uncommitted > 0:
            base_score -= min(uncommitted * 0.1, 0.3)
        
        # Penalize untracked files
        if untracked > 0:
            base_score -= min(untracked * 0.05, 0.2)
        
        # Penalize being behind remote
        if ahead_behind[1] > 0:
            base_score -= min(ahead_behind[1] * 0.05, 0.2)
        
        return max(base_score, 0.0)
    
    async def detect_code_changes(self) -> List[CodeChange]:
        """
        Detect and analyze code changes in the repository
        """
        # AI REASONING: Code change detection and analysis
        # PSEUDOCODE:
        # 1. Scan repository for modified, added, and deleted files
        # 2. Analyze change patterns and complexity
        # 3. Identify dependencies and affected components
        # 4. Assess change impact and risk level
        # 5. Categorize changes by type and priority
        # 6. Generate change descriptions and recommendations
        # 7. Track change history and patterns
        # 8. Validate change consistency and quality
        
        changes = []
        
        try:
            # Get modified files
            for diff in self.repo.index.diff(None):
                change = CodeChange(
                    file_path=diff.a_path,
                    change_type='modify',
                    description=f"Modified {diff.a_path}",
                    priority=self._assess_change_priority(diff),
                    complexity=self._assess_change_complexity(diff),
                    estimated_effort=self._estimate_change_effort(diff),
                    dependencies=self._identify_dependencies(diff.a_path)
                )
                changes.append(change)
            
            # Get untracked files
            for file_path in self.repo.untracked_files:
                change = CodeChange(
                    file_path=file_path,
                    change_type='add',
                    description=f"New file: {file_path}",
                    priority=5,
                    complexity='low',
                    estimated_effort=1,
                    dependencies=[]
                )
                changes.append(change)
            
            # Get deleted files
            for diff in self.repo.index.diff(None):
                if diff.deleted_file:
                    change = CodeChange(
                        file_path=diff.a_path,
                        change_type='delete',
                        description=f"Deleted {diff.a_path}",
                        priority=3,
                        complexity='low',
                        estimated_effort=1,
                        dependencies=[]
                    )
                    changes.append(change)
            
            # Update change history
            self.change_history.extend(changes)
            
            return changes
            
        except Exception as e:
            logger.error(f"Error detecting code changes: {e}")
            return []
    
    def _assess_change_priority(self, diff) -> int:
        """
        Assess the priority of a code change
        """
        # AI REASONING: Change priority assessment
        # PSEUDOCODE:
        # 1. Analyze file type and criticality
        # 2. Assess change size and complexity
        # 3. Consider file usage and dependencies
        # 4. Evaluate potential impact on system stability
        # 5. Factor in business criticality and urgency
        # 6. Generate priority score (1-10)
        
        priority = 5  # Default priority
        
        # Higher priority for critical files
        critical_files = ['requirements.txt', 'setup.py', 'config.py', 'main.py']
        if diff.a_path in critical_files:
            priority += 3
        
        # Higher priority for larger changes
        if hasattr(diff, 'stats') and diff.stats:
            total_changes = sum(diff.stats.values())
            if total_changes > 100:
                priority += 2
            elif total_changes > 50:
                priority += 1
        
        # Higher priority for Python files
        if diff.a_path.endswith('.py'):
            priority += 1
        
        return min(priority, 10)
    
    def _assess_change_complexity(self, diff) -> str:
        """
        Assess the complexity of a code change
        """
        # AI REASONING: Change complexity assessment
        # PSEUDOCODE:
        # 1. Analyze change size and scope
        # 2. Evaluate code structure modifications
        # 3. Assess algorithm and logic complexity
        # 4. Consider architectural impact
        # 5. Factor in testing requirements
        # 6. Generate complexity classification
        
        if hasattr(diff, 'stats') and diff.stats:
            total_changes = sum(diff.stats.values())
            if total_changes > 200:
                return 'high'
            elif total_changes > 50:
                return 'medium'
        
        return 'low'
    
    def _estimate_change_effort(self, diff) -> int:
        """
        Estimate effort required for a code change
        """
        # AI REASONING: Effort estimation
        # PSEUDOCODE:
        # 1. Analyze change size and complexity
        # 2. Consider file type and modification type
        # 3. Factor in testing and validation requirements
        # 4. Assess documentation and review needs
        # 5. Generate effort estimate in hours
        
        base_effort = 1
        
        if hasattr(diff, 'stats') and diff.stats:
            total_changes = sum(diff.stats.values())
            base_effort = max(1, total_changes // 50)
        
        # Add effort for complex files
        if diff.a_path.endswith('.py'):
            base_effort += 1
        
        return base_effort
    
    def _identify_dependencies(self, file_path: str) -> List[str]:
        """
        Identify dependencies for a file
        """
        # AI REASONING: Dependency analysis
        # PSEUDOCODE:
        # 1. Parse file imports and dependencies
        # 2. Analyze function and class dependencies
        # 3. Identify external library dependencies
        # 4. Map internal module dependencies
        # 5. Generate dependency graph
        
        dependencies = []
        
        try:
            if file_path.endswith('.py'):
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Simple import detection
                import_lines = [line.strip() for line in content.split('\n') 
                              if line.strip().startswith(('import ', 'from '))]
                dependencies.extend(import_lines)
                
        except Exception as e:
            logger.warning(f"Error identifying dependencies for {file_path}: {e}")
        
        return dependencies
    
    async def commit_changes(self, commit_message: str, files: List[str] = None) -> bool:
        """
        Commit changes to the repository
        """
        # AI REASONING: Change commitment and validation
        # PSEUDOCODE:
        # 1. Validate changes before committing
        # 2. Check for conflicts and merge issues
        # 3. Run pre-commit tests and validations
        # 4. Generate appropriate commit message
        # 5. Stage and commit changes
        # 6. Validate commit success and integrity
        # 7. Update change tracking and history
        # 8. Generate commit summary and notifications
        
        try:
            # Stage changes
            if files:
                for file_path in files:
                    self.repo.index.add([file_path])
            else:
                self.repo.index.add('*')
            
            # Check if there are changes to commit
            if not self.repo.index.diff('HEAD'):
                logger.info("No changes to commit")
                return True
            
            # Create commit
            commit = self.repo.index.commit(commit_message)
            logger.info(f"Committed changes: {commit.hexsha[:8]} - {commit_message}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error committing changes: {e}")
            return False
    
    async def push_changes(self, remote: str = 'origin', branch: str = None) -> bool:
        """
        Push changes to remote repository
        """
        # AI REASONING: Remote synchronization
        # PSEUDOCODE:
        # 1. Validate local changes and commit status
        # 2. Check remote connectivity and permissions
        # 3. Resolve conflicts and merge issues
        # 4. Push changes to remote repository
        # 5. Validate push success and remote sync
        # 6. Update remote tracking information
        # 7. Generate push summary and notifications
        
        try:
            if not branch:
                branch = self.repo.active_branch.name
            
            # Push to remote
            self.repo.git.push(remote, branch)
            logger.info(f"Pushed changes to {remote}/{branch}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error pushing changes: {e}")
            return False
    
    async def create_branch(self, branch_name: str, base_branch: str = None) -> bool:
        """
        Create a new branch
        """
        # AI REASONING: Branch management
        # PSEUDOCODE:
        # 1. Validate branch name and naming conventions
        # 2. Check for existing branches and conflicts
        # 3. Create branch from appropriate base
        # 4. Set up branch tracking and protection
        # 5. Update branch metadata and documentation
        # 6. Generate branch creation summary
        
        try:
            if not base_branch:
                base_branch = self.repo.active_branch.name
            
            # Create new branch
            new_branch = self.repo.create_head(branch_name, base_branch)
            logger.info(f"Created branch: {branch_name} from {base_branch}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating branch: {e}")
            return False
    
    async def merge_branch(self, source_branch: str, target_branch: str = None) -> bool:
        """
        Merge a branch into target branch
        """
        # AI REASONING: Branch merging and conflict resolution
        # PSEUDOCODE:
        # 1. Validate source and target branches
        # 2. Check for merge conflicts and issues
        # 3. Perform pre-merge validation and testing
        # 4. Execute merge operation
        # 5. Resolve conflicts if necessary
        # 6. Validate merge success and integrity
        # 7. Update branch status and tracking
        # 8. Generate merge summary and notifications
        
        try:
            if not target_branch:
                target_branch = self.repo.active_branch.name
            
            # Switch to target branch
            self.repo.git.checkout(target_branch)
            
            # Merge source branch
            self.repo.git.merge(source_branch)
            logger.info(f"Merged {source_branch} into {target_branch}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error merging branch: {e}")
            return False
    
    async def run_repository_maintenance(self) -> Dict[str, Any]:
        """
        Run repository maintenance tasks
        """
        # AI REASONING: Repository maintenance orchestration
        # PSEUDOCODE:
        # 1. Analyze repository health and status
        # 2. Identify maintenance tasks and priorities
        # 3. Execute garbage collection and optimization
        # 4. Update dependencies and security patches
        # 5. Clean up old branches and tags
        # 6. Validate repository integrity
        # 7. Generate maintenance report and recommendations
        # 8. Update repository metadata and statistics
        
        logger.info("Starting repository maintenance")
        
        maintenance_results = {
            "timestamp": datetime.now().isoformat(),
            "tasks_completed": [],
            "issues_found": [],
            "recommendations": []
        }
        
        try:
            # Get current status
            status = await self.get_repository_status()
            
            # Run garbage collection
            if status.health_score < 0.8:
                self.repo.git.gc()
                maintenance_results["tasks_completed"].append("garbage_collection")
            
            # Check for security updates
            security_issues = await self._check_security_updates()
            if security_issues:
                maintenance_results["issues_found"].extend(security_issues)
            
            # Generate recommendations
            recommendations = await self._generate_maintenance_recommendations(status)
            maintenance_results["recommendations"].extend(recommendations)
            
            logger.info("Repository maintenance completed")
            return maintenance_results
            
        except Exception as e:
            logger.error(f"Error in repository maintenance: {e}")
            maintenance_results["issues_found"].append(f"Maintenance error: {e}")
            return maintenance_results
    
    async def _check_security_updates(self) -> List[str]:
        """
        Check for security updates and vulnerabilities
        """
        # AI REASONING: Security assessment
        # PSEUDOCODE:
        # 1. Scan dependencies for known vulnerabilities
        # 2. Check for outdated packages and libraries
        # 3. Analyze code for security issues
        # 4. Validate authentication and authorization
        # 5. Generate security recommendations
        
        issues = []
        
        # Check requirements.txt for outdated packages
        requirements_file = Path(self.repo_path) / "requirements.txt"
        if requirements_file.exists():
            try:
                # Simple check for common security issues
                with open(requirements_file, 'r') as f:
                    content = f.read()
                    if 'requests<2.25.0' in content:
                        issues.append("Outdated requests library - security vulnerability")
                    if 'urllib3<1.26.0' in content:
                        issues.append("Outdated urllib3 library - security vulnerability")
            except Exception as e:
                logger.warning(f"Error checking requirements.txt: {e}")
        
        return issues
    
    async def _generate_maintenance_recommendations(self, status: RepositoryStatus) -> List[str]:
        """
        Generate maintenance recommendations
        """
        # AI REASONING: Recommendation generation
        # PSEUDOCODE:
        # 1. Analyze repository status and health metrics
        # 2. Identify improvement opportunities
        # 3. Prioritize recommendations by impact and effort
        # 4. Generate actionable recommendations
        # 5. Consider best practices and industry standards
        
        recommendations = []
        
        if status.uncommitted_changes > 0:
            recommendations.append("Commit pending changes to improve repository health")
        
        if status.untracked_files > 10:
            recommendations.append("Review and organize untracked files")
        
        if status.ahead_behind[1] > 0:
            recommendations.append("Pull latest changes from remote repository")
        
        if status.health_score < 0.7:
            recommendations.append("Run repository optimization and cleanup")
        
        return recommendations
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info(f"{self.name} cleanup completed")

# Example usage
async def main():
    config = {
        "repo_path": ".",
        "auto_commit": True,
        "auto_push": False,
        "branch_protection": True
    }
    
    agent = RepositoryManagementAgent(config)
    await agent.initialize()
    
    try:
        # Get repository status
        status = await agent.get_repository_status()
        print(f"Repository health: {status.health_score:.2f}")
        print(f"Uncommitted changes: {status.uncommitted_changes}")
        
        # Detect changes
        changes = await agent.detect_code_changes()
        print(f"Detected {len(changes)} changes")
        
        # Run maintenance
        maintenance = await agent.run_repository_maintenance()
        print(f"Maintenance completed: {len(maintenance['tasks_completed'])} tasks")
        
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 