"""
State definitions for ReproAgent.
Complete state tracking across all reproduction phases.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np


class Phase(Enum):
    """Workflow phases."""
    IDLE = "idle"
    PARSING = "parsing"
    REPO_ANALYSIS = "repo_analysis"
    SETUP = "setup"
    EXECUTION = "execution"
    DEBUGGING = "debugging"
    EXPERIMENTATION = "experimentation"
    COMPARISON = "comparison"
    COMPLETE = "complete"


class DifficultyLevel(Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class PaperState:
    """Paper understanding state."""
    pdf_path: str = ""
    title: str = ""
    abstract: str = ""
    dataset: str = ""
    model: str = ""
    target_metric: float = 0.0
    metric_name: str = "accuracy"
    github_links: List[str] = field(default_factory=list)
    key_claims: List[str] = field(default_factory=list)
    parsed: bool = False
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'dataset': self.dataset,
            'model': self.model,
            'target_metric': self.target_metric,
            'github_links': self.github_links,
            'parsed': self.parsed
        }


@dataclass
class RepoState:
    """Repository state."""
    url: str = ""
    cloned: bool = False
    local_path: str = ""
    readme_content: str = ""
    setup_instructions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    entry_point: str = ""
    framework: str = ""
    repo_quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'cloned': self.cloned,
            'framework': self.framework,
            'entry_point': self.entry_point,
            'dependencies': len(self.dependencies)
        }


@dataclass
class EnvironmentState:
    """Execution environment state."""
    python_version: str = "3.10"
    cuda_available: bool = False
    packages_installed: List[str] = field(default_factory=list)
    setup_complete: bool = False
    setup_errors: List[str] = field(default_factory=list)


@dataclass
class ExecutionState:
    """Code execution state."""
    current_phase: str = "idle"
    commands_run: List[str] = field(default_factory=list)
    last_command: str = ""
    last_output: str = ""
    last_error: str = ""
    process_running: bool = False
    logs: List[str] = field(default_factory=list)


@dataclass
class DebugState:
    """Debugging state."""
    errors_encountered: List[Dict[str, str]] = field(default_factory=list)
    current_error: str = ""
    error_type: str = ""
    fix_attempts: List[Dict[str, Any]] = field(default_factory=list)
    solutions_tried: List[str] = field(default_factory=list)
    last_hypothesis: str = ""
    debugging_level: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_count': len(self.errors_encountered),
            'current_error': self.current_error,
            'fix_attempts': len(self.fix_attempts),
            'debugging_level': self.debugging_level
        }


@dataclass
class ExperimentState:
    """Experimentation state."""
    current_config: Dict[str, Any] = field(default_factory=dict)
    experiments_run: int = 0
    best_metric: float = 0.0
    current_metric: float = 0.0
    target_metric: float = 0.0
    gap: float = 0.0
    configs_tried: List[Dict] = field(default_factory=list)
    improvement_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiments_run': self.experiments_run,
            'current_metric': self.current_metric,
            'best_metric': self.best_metric,
            'target_metric': self.target_metric,
            'gap': self.gap
        }


@dataclass
class ReasoningState:
    """Agent reasoning state."""
    current_hypothesis: str = ""
    hypotheses_formed: List[str] = field(default_factory=list)
    active_strategy: str = "systematic"
    confidence_score: float = 0.5
    next_action_plan: List[str] = field(default_factory=list)
    learned_insights: List[str] = field(default_factory=list)


@dataclass
class MetaState:
    """Meta information."""
    step_count: int = 0
    phase: Phase = Phase.IDLE
    success: bool = False
    failure_reason: str = ""
    difficulty_level: DifficultyLevel = DifficultyLevel.EASY
    agent_mode: str = "exploration"
    time_budget_used: float = 0.0


@dataclass
class ReproductionState:
    """
    Complete reproduction state.
    This is the full state the environment tracks.
    """
    paper: PaperState = field(default_factory=PaperState)
    repo: RepoState = field(default_factory=RepoState)
    environment: EnvironmentState = field(default_factory=EnvironmentState)
    execution: ExecutionState = field(default_factory=ExecutionState)
    debug: DebugState = field(default_factory=DebugState)
    experiment: ExperimentState = field(default_factory=ExperimentState)
    reasoning: ReasoningState = field(default_factory=ReasoningState)
    meta: MetaState = field(default_factory=MetaState)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'paper': self.paper.to_dict(),
            'repo': self.repo.to_dict(),
            'debug': self.debug.to_dict(),
            'experiment': self.experiment.to_dict(),
            'meta': {
                'step_count': self.meta.step_count,
                'phase': self.meta.phase.value,
                'success': self.meta.success,
                'difficulty_level': self.meta.difficulty_level.value
            }
        }
    
    def to_observation(self) -> Dict[str, np.ndarray]:
        """
        Convert to Gymnasium observation.
        Fixed-size numpy arrays for RL compatibility.
        """
        return {
            # Paper features (5 dims)
            'paper_features': np.clip(np.array([
                float(self.paper.parsed),
                self.paper.confidence,
                self.paper.target_metric,
                len(self.paper.github_links),
                len(self.paper.key_claims) / 10.0
            ], dtype=np.float32), 0.0, 1.0),
            
            # Repo features (5 dims)
            'repo_features': np.clip(np.array([
                float(self.repo.cloned),
                self.repo.repo_quality_score,
                len(self.repo.dependencies) / 50.0,
                float(bool(self.repo.entry_point)),
                float(self.environment.setup_complete)
            ], dtype=np.float32), 0.0, 1.0),
            
            # Execution features (5 dims)
            'execution_features': np.clip(np.array([
                float(self.execution.process_running),
                self.debug.debugging_level / 5.0,
                len(self.debug.errors_encountered) / 10.0,
                len(self.execution.commands_run) / 20.0,
                float(bool(self.execution.last_error))
            ], dtype=np.float32), 0.0, 1.0),
            
            # Experiment features (5 dims)
            'experiment_features': np.clip(np.array([
                self.experiment.current_metric,
                self.experiment.best_metric,
                self.experiment.target_metric,
                self.experiment.gap,
                self.experiment.experiments_run / 50.0
            ], dtype=np.float32), 0.0, 1.0),
            
            # Meta features (5 dims)
            'meta_features': np.clip(np.array([
                self.meta.step_count / 100.0,
                {'easy': 0.33, 'medium': 0.66, 'hard': 1.0}[self.meta.difficulty_level.value],
                self.meta.time_budget_used,
                float(self.meta.success),
                {'exploration': 0.0, 'exploitation': 1.0}.get(self.meta.agent_mode, 0.5)
            ], dtype=np.float32), 0.0, 1.0)
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "="*60,
            "📊 REPRODUCTION STATE",
            "="*60,
            f"Phase: {self.meta.phase.value} | Step: {self.meta.step_count}",
            ""
        ]
        
        if self.paper.parsed:
            lines.extend([
                f"📄 Paper: {self.paper.title[:50]}",
                f"   Target: {self.paper.target_metric:.3f} {self.paper.metric_name}",
                f"   Dataset: {self.paper.dataset}",
                ""
            ])
        
        if self.repo.cloned:
            lines.extend([
                f"📦 Repo: {self.repo.url[:50]}",
                f"   Framework: {self.repo.framework}",
                ""
            ])
        
        if self.experiment.experiments_run > 0:
            lines.extend([
                f"🧪 Experiments: {self.experiment.experiments_run}",
                f"   Current: {self.experiment.current_metric:.3f}",
                f"   Best: {self.experiment.best_metric:.3f}",
                f"   Gap: {self.experiment.gap:.3f}",
                ""
            ])
        
        if self.debug.errors_encountered:
            lines.extend([
                f"🐛 Errors: {len(self.debug.errors_encountered)}",
                f"   Fixes: {len(self.debug.fix_attempts)}",
                ""
            ])
        
        lines.append("="*60)
        return "\n".join(lines)
