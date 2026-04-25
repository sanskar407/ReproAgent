"""
Baseline agents for comparison.
Provides random and simple heuristic baselines.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, Any

from reproagent.environment import ReproAgentEnv


class RandomBaseline:
    """Random action baseline."""
    
    def __init__(self, env: ReproAgentEnv):
        self.env = env
    
    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """Select random action."""
        return self.env.action_space.sample()
    
    def reset(self):
        """Reset agent."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats."""
        return {'type': 'random'}


class PhaseBaseline:
    """
    Phase-based heuristic baseline.
    Follows fixed strategy per phase.
    """
    
    def __init__(self, env: ReproAgentEnv):
        self.env = env
        self.action_space = env.action_space_helper
        self.phase_sequence = [
            'PARSE_PDF',
            'EXTRACT_GITHUB',
            'CLONE_REPO',
            'READ_README',
            'INSTALL_REQUIREMENTS',
            'RUN_TRAINING',
            'RUN_EXPERIMENT'
        ]
        self.current_index = 0
    
    def select_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """Select action based on phase sequence."""
        
        if self.current_index < len(self.phase_sequence):
            action_name = self.phase_sequence[self.current_index]
            
            # Find action ID
            from reproagent.actions import ActionType
            
            try:
                action_type = ActionType[action_name]
                action_id = self.action_space.get_id_by_action(action_type)
                self.current_index += 1
                return action_id
            except:
                return self.env.action_space.sample()
        else:
            # After sequence, random
            return self.env.action_space.sample()
    
    def reset(self):
        """Reset agent."""
        self.current_index = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats."""
        return {'type': 'phase_based'}


def evaluate_baseline(baseline_class, env: ReproAgentEnv, num_episodes: int = 5):
    """
    Evaluate baseline agent.
    
    Args:
        baseline_class: Baseline class
        env: Environment
        num_episodes: Number of episodes
        
    Returns:
        Results dict
    """
    
    agent = baseline_class(env)
    
    results = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        agent.reset()
        
        episode_reward = 0
        steps = 0
        
        for _ in range(env.max_steps):
            action = agent.select_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        results.append({
            'reward': episode_reward,
            'steps': steps,
            'final_metric': info.get('current_metric', 0.0),
            'success': info.get('success', False)
        })
    
    # Calculate statistics
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_metric = np.mean([r['final_metric'] for r in results])
    success_rate = np.mean([r['success'] for r in results])
    
    return {
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'avg_metric': avg_metric,
        'success_rate': success_rate,
        'results': results
    }


def compare_baselines():
    """Compare all baseline agents."""
    
    print("="*70)
    print("📊 BASELINE COMPARISON")
    print("="*70)
    print()
    
    env = ReproAgentEnv(difficulty="easy", max_steps=30, use_llm=False)
    
    baselines = {
        'Random': RandomBaseline,
        'Phase-Based': PhaseBaseline
    }
    
    results = {}
    
    for name, baseline_class in baselines.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_baseline(baseline_class, env, num_episodes=5)
        print(f"  Avg Metric: {results[name]['avg_metric']:.3f}")
        print(f"  Success Rate: {results[name]['success_rate']*100:.1f}%")
        print()
    
    # Print comparison table
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Baseline':<15} {'Avg Metric':<15} {'Success Rate':<15} {'Avg Steps':<15}")
    print("-"*70)
    
    for name, result in results.items():
        print(f"{name:<15} {result['avg_metric']:<15.3f} {result['success_rate']*100:<14.1f}% {result['avg_steps']:<15.1f}")
    
    print("="*70)


if __name__ == "__main__":
    compare_baselines()
