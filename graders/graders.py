"""
Grading system for ReproAgent (OpenEnv requirement).
Evaluates agent performance on reproduction tasks.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from pathlib import Path

from reproagent.environment import ReproAgentEnv
from agents.reasoning_agent import create_agent


class ReproductionGrader:
    """
    Grades agent performance on paper reproduction tasks.
    
    Metrics:
    1. Success rate (reached target)
    2. Efficiency (steps taken)
    3. Quality (final metric value)
    4. Robustness (across difficulties)
    """
    
    def __init__(self):
        self.results = []
    
    def grade_agent(
        self,
        agent,
        env: ReproAgentEnv,
        num_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Grade agent performance.
        
        Args:
            agent: Agent to evaluate
            env: Environment
            num_episodes: Number of episodes
            
        Returns:
            Grading report
        """
        print(f"📊 Grading agent over {num_episodes} episodes...")
        
        episode_results = []
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            obs, info = env.reset()
            agent.reset()
            
            episode_reward = 0
            steps = 0
            
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action = agent.select_action(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
            
            result = {
                'episode': episode + 1,
                'success': terminated,
                'steps': steps,
                'reward': episode_reward,
                'final_metric': info.get('current_metric', 0.0),
                'target_metric': info.get('target_metric', 0.0),
                'gap': info.get('gap', 0.0)
            }
            
            episode_results.append(result)
            
            print(f"  Success: {result['success']}")
            print(f"  Metric: {result['final_metric']:.3f} / {result['target_metric']:.3f}")
            print(f"  Steps: {result['steps']}")
        
        # Calculate statistics
        grade = self._calculate_grade(episode_results)
        
        self.results.append(grade)
        
        return grade
    
    def _calculate_grade(self, episode_results: List[Dict]) -> Dict[str, Any]:
        """Calculate overall grade from episode results."""
        
        success_count = sum(1 for r in episode_results if r['success'])
        success_rate = success_count / len(episode_results)
        
        avg_steps = np.mean([r['steps'] for r in episode_results])
        avg_reward = np.mean([r['reward'] for r in episode_results])
        avg_final_metric = np.mean([r['final_metric'] for r in episode_results])
        avg_gap = np.mean([r['gap'] for r in episode_results])
        
        # Calculate letter grade
        if success_rate >= 0.9:
            letter_grade = 'A'
        elif success_rate >= 0.7:
            letter_grade = 'B'
        elif success_rate >= 0.5:
            letter_grade = 'C'
        elif success_rate >= 0.3:
            letter_grade = 'D'
        else:
            letter_grade = 'F'
        
        # Calculate efficiency score
        efficiency = 1.0 - (avg_steps / 100.0)  # Normalized
        
        # Calculate quality score
        quality = avg_final_metric
        
        # Overall score (weighted average)
        overall_score = (
            success_rate * 0.5 +
            efficiency * 0.2 +
            quality * 0.3
        ) * 100
        
        return {
            'num_episodes': len(episode_results),
            'success_rate': success_rate,
            'success_count': success_count,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward,
            'avg_final_metric': avg_final_metric,
            'avg_gap': avg_gap,
            'efficiency_score': efficiency,
            'quality_score': quality,
            'overall_score': overall_score,
            'letter_grade': letter_grade,
            'episode_results': episode_results
        }
    
    def generate_report(self, grade: Dict[str, Any]) -> str:
        """Generate human-readable grading report."""
        
        lines = [
            "="*60,
            "📊 REPRODUCTION GRADING REPORT",
            "="*60,
            "",
            f"Episodes Evaluated: {grade['num_episodes']}",
            "",
            "--- PERFORMANCE METRICS ---",
            "",
            f"Success Rate:     {grade['success_rate']*100:.1f}% ({grade['success_count']}/{grade['num_episodes']})",
            f"Avg Final Metric: {grade['avg_final_metric']:.3f}",
            f"Avg Gap:          {grade['avg_gap']:.3f}",
            f"Avg Steps:        {grade['avg_steps']:.1f}",
            f"Avg Reward:       {grade['avg_reward']:.2f}",
            "",
            "--- SCORES ---",
            "",
            f"Efficiency Score: {grade['efficiency_score']*100:.1f}/100",
            f"Quality Score:    {grade['quality_score']*100:.1f}/100",
            f"Overall Score:    {grade['overall_score']:.1f}/100",
            "",
            f"GRADE: {grade['letter_grade']}",
            "",
            "="*60
        ]
        
        return "\n".join(lines)
    
    def compare_agents(
        self,
        agents: List[Tuple[str, Any]],
        env: ReproAgentEnv,
        num_episodes: int = 5
    ) -> Dict[str, Any]:
        """
        Compare multiple agents.
        
        Args:
            agents: List of (name, agent) tuples
            env: Environment
            num_episodes: Episodes per agent
            
        Returns:
            Comparison report
        """
        print(f"📊 Comparing {len(agents)} agents...")
        
        results = {}
        
        for name, agent in agents:
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print(f"{'='*60}")
            
            grade = self.grade_agent(agent, env, num_episodes)
            results[name] = grade
        
        # Generate comparison
        comparison = self._generate_comparison(results)
        
        return comparison
    
    def _generate_comparison(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comparison between agents."""
        
        comparison = {
            'agents': list(results.keys()),
            'metrics': {}
        }
        
        # Compare each metric
        for metric in ['success_rate', 'avg_steps', 'avg_final_metric', 'overall_score']:
            comparison['metrics'][metric] = {
                agent: results[agent][metric]
                for agent in results
            }
        
        # Find best agent
        best_agent = max(results.items(), key=lambda x: x[1]['overall_score'])
        comparison['best_agent'] = best_agent[0]
        comparison['best_score'] = best_agent[1]['overall_score']
        
        return comparison
    
    def print_comparison(self, comparison: Dict[str, Any]):
        """Print comparison table."""
        
        print("\n" + "="*80)
        print("📊 AGENT COMPARISON")
        print("="*80)
        
        # Header
        agents = comparison['agents']
        print(f"{'Metric':<25} " + " ".join(f"{a:<15}" for a in agents))
        print("-"*80)
        
        # Metrics
        metrics = comparison['metrics']
        
        metric_names = {
            'success_rate': 'Success Rate',
            'avg_steps': 'Avg Steps',
            'avg_final_metric': 'Avg Metric',
            'overall_score': 'Overall Score'
        }
        
        for metric_key, metric_name in metric_names.items():
            values = metrics[metric_key]
            
            if metric_key == 'success_rate':
                row = f"{metric_name:<25} " + " ".join(
                    f"{values[a]*100:>14.1f}%" for a in agents
                )
            elif metric_key == 'overall_score':
                row = f"{metric_name:<25} " + " ".join(
                    f"{values[a]:>14.1f}/100" for a in agents
                )
            else:
                row = f"{metric_name:<25} " + " ".join(
                    f"{values[a]:>15.3f}" for a in agents
                )
            
            print(row)
        
        print("="*80)
        print(f"🏆 Best Agent: {comparison['best_agent']} (Score: {comparison['best_score']:.1f})")
        print("="*80)


# Test function
def test_grader():
    """Test the grading system."""
    
    from reproagent.environment import ReproAgentEnv
    from agents.reasoning_agent import create_agent
    
    # Create environment
    env = ReproAgentEnv(difficulty="easy", max_steps=30, use_llm=False)
    
    # Create agents
    reasoning_agent = create_agent(env, "reasoning", use_llm=False)
    random_agent = create_agent(env, "random")
    
    # Grade and compare
    grader = ReproductionGrader()
    
    comparison = grader.compare_agents(
        [
            ("Reasoning Agent", reasoning_agent),
            ("Random Agent", random_agent)
        ],
        env,
        num_episodes=3
    )
    
    grader.print_comparison(comparison)


if __name__ == "__main__":
    test_grader()
