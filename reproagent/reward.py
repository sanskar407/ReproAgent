"""
Reward function for ReproAgent.
Incentivizes successful paper reproduction.
"""

from dataclasses import dataclass
from typing import Dict, Any
from reproagent.state import ReproductionState, Phase


@dataclass
class RewardComponents:
    """Breakdown of reward calculation."""
    
    progress_reward: float = 0.0
    metric_reward: float = 0.0
    efficiency_penalty: float = 0.0
    error_penalty: float = 0.0
    success_bonus: float = 0.0
    total_reward: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'progress': self.progress_reward,
            'metric': self.metric_reward,
            'efficiency': self.efficiency_penalty,
            'error': self.error_penalty,
            'success': self.success_bonus,
            'total': self.total_reward
        }


class RewardFunction:
    """
    Calculates rewards based on reproduction progress.
    
    Reward structure:
    1. Progress rewards (phase completion)
    2. Metric rewards (getting closer to target)
    3. Efficiency penalties (too many steps/errors)
    4. Success bonus (reproduction complete)
    """
    
    def __init__(
        self,
        max_steps: int = 100,
        target_metric: float = 0.95
    ):
        self.max_steps = max_steps
        self.target_metric = target_metric
        
        # Reward weights
        self.progress_weight = 20.0
        self.metric_weight = 50.0
        self.efficiency_weight = 0.5
        self.error_weight = 5.0
        self.success_bonus_value = 100.0
        
        # Phase completion rewards
        self.phase_rewards = {
            Phase.PARSING: 10.0,
            Phase.REPO_ANALYSIS: 15.0,
            Phase.SETUP: 20.0,
            Phase.EXECUTION: 25.0,
            Phase.EXPERIMENTATION: 30.0,
            Phase.COMPARISON: 40.0,
            Phase.COMPLETE: 50.0
        }
    
    def calculate_reward(
        self,
        prev_state: ReproductionState,
        action: Any,
        new_state: ReproductionState
    ) -> RewardComponents:
        """
        Calculate reward for state transition.
        
        Args:
            prev_state: Previous state
            action: Action taken
            new_state: New state after action
            
        Returns:
            RewardComponents with breakdown
        """
        components = RewardComponents()
        
        # 1. Progress reward (phase advancement)
        components.progress_reward = self._progress_reward(prev_state, new_state)
        
        # 2. Metric reward (closer to target)
        components.metric_reward = self._metric_reward(prev_state, new_state)
        
        # 3. Efficiency penalty
        components.efficiency_penalty = self._efficiency_penalty(new_state)
        
        # 4. Error penalty
        components.error_penalty = self._error_penalty(prev_state, new_state)
        
        # 5. Success bonus
        if new_state.meta.success:
            components.success_bonus = self._success_bonus(new_state)
        
        # Total raw reward calculation
        raw_total = (
            components.progress_reward +
            components.metric_reward -
            components.efficiency_penalty -
            components.error_penalty +
            components.success_bonus
        )
        
        # Normalize total reward between 0.0 and 1.0
        # Theoretical max is roughly ~300-400 (Phases=~190, Success Bonus=~100-200, Metric=~50)
        max_expected_reward = 400.0
        components.total_reward = max(0.0, min(1.0, raw_total / max_expected_reward))
        
        return components
    
    def _progress_reward(
        self,
        prev_state: ReproductionState,
        new_state: ReproductionState
    ) -> float:
        """Reward for progressing through phases."""
        reward = 0.0
        
        # Phase completion
        if new_state.meta.phase != prev_state.meta.phase:
            reward += self.phase_rewards.get(new_state.meta.phase, 0.0)
        
        # Milestone rewards
        if not prev_state.paper.parsed and new_state.paper.parsed:
            reward += 10.0
        
        if not prev_state.repo.cloned and new_state.repo.cloned:
            reward += 15.0
        
        if not prev_state.environment.setup_complete and new_state.environment.setup_complete:
            reward += 20.0
        
        return reward * self.progress_weight / 20.0
    
    def _metric_reward(
        self,
        prev_state: ReproductionState,
        new_state: ReproductionState
    ) -> float:
        """Reward for getting closer to target metric."""
        if new_state.experiment.target_metric == 0:
            return 0.0
        
        # Improvement in metric
        improvement = new_state.experiment.current_metric - prev_state.experiment.current_metric
        
        if improvement > 0:
            # Normalize improvement
            normalized = improvement / new_state.experiment.target_metric
            return normalized * self.metric_weight
        
        return 0.0
    
    def _efficiency_penalty(self, state: ReproductionState) -> float:
        """Penalize inefficiency (too many steps)."""
        step_ratio = state.meta.step_count / self.max_steps
        return step_ratio * self.efficiency_weight
    
    def _error_penalty(
        self,
        prev_state: ReproductionState,
        new_state: ReproductionState
    ) -> float:
        """Penalize errors."""
        new_errors = len(new_state.debug.errors_encountered) - len(prev_state.debug.errors_encountered)
        return new_errors * self.error_weight
    
    def _success_bonus(self, state: ReproductionState) -> float:
        """Large bonus for successful reproduction."""
        bonus = self.success_bonus_value
        
        # Efficiency multiplier
        efficiency = 1.0 - (state.meta.step_count / self.max_steps)
        bonus *= (0.5 + 0.5 * efficiency)
        
        # Difficulty multiplier
        difficulty_multipliers = {
            'easy': 1.0,
            'medium': 1.5,
            'hard': 2.0
        }
        bonus *= difficulty_multipliers.get(state.meta.difficulty_level.value, 1.0)
        
        return bonus
    
    def get_reward_summary(self, components: RewardComponents) -> str:
        """Generate human-readable summary."""
        lines = [
            "="*50,
            "💰 REWARD BREAKDOWN",
            "="*50,
            f"Progress:   +{components.progress_reward:7.2f}",
            f"Metric:     +{components.metric_reward:7.2f}",
            f"Efficiency: -{components.efficiency_penalty:7.2f}",
            f"Errors:     -{components.error_penalty:7.2f}",
        ]
        
        if components.success_bonus > 0:
            lines.append(f"Success:    +{components.success_bonus:7.2f}")
        
        lines.extend([
            "-"*50,
            f"TOTAL:       {components.total_reward:7.2f}",
            "="*50
        ])
        
        return "\n".join(lines)
