"""
Inference script for running trained/deployed agent.
Usage: python inference.py --difficulty easy --steps 30
"""

import argparse
import sys
from pathlib import Path

from reproagent.environment import ReproAgentEnv
from agents.reasoning_agent import create_agent


def run_inference(
    difficulty: str = "easy",
    agent_type: str = "reasoning",
    max_steps: int = 30,
    use_llm: bool = False,
    verbose: bool = True
):
    """
    Run inference with agent.
    
    Args:
        difficulty: Difficulty level
        agent_type: Agent type
        max_steps: Maximum steps
        use_llm: Use LLM for reasoning
        verbose: Print detailed logs
    """
    
    if verbose:
        print("="*70)
        print("🚀 REPROAGENT INFERENCE")
        print("="*70)
        print(f"Difficulty: {difficulty}")
        print(f"Agent: {agent_type}")
        print(f"Max Steps: {max_steps}")
        print(f"LLM: {'Enabled' if use_llm else 'Disabled'}")
        print("="*70)
        print()
    
    # Create environment
    env = ReproAgentEnv(
        difficulty=difficulty,
        max_steps=max_steps,
        use_llm=use_llm,
        render_mode='human' if verbose else None
    )
    
    # Create agent
    agent = create_agent(env, agent_type, use_llm=use_llm)
    
    # Run episode
    obs, info = env.reset()
    agent.reset()
    
    total_reward = 0
    step = 0
    
    if verbose:
        print("\n🎬 Starting episode...\n")
    
    while step < max_steps:
        # Select action
        action = agent.select_action(obs, info)
        
        # Get reasoning
        reasoning = agent.get_reasoning(env.state, action)
        
        if verbose:
            print(f"Step {step + 1}: {reasoning}")
        
        # Execute
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step += 1
        
        if verbose:
            print(f"  Reward: {reward:.2f} | Metric: {info.get('current_metric', 0.0):.3f}")
            print()
        
        if terminated or truncated:
            break
    
    # Results
    final_metric = info.get('current_metric', 0.0)
    target_metric = info.get('target_metric', 0.0)
    success = info.get('success', False)
    
    if verbose:
        print("="*70)
        print("📊 RESULTS")
        print("="*70)
        print(f"Steps: {step}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Metric: {final_metric:.3f}")
        print(f"Target Metric: {target_metric:.3f}")
        print(f"Gap: {target_metric - final_metric:.3f}")
        print(f"Success: {'✅ YES' if success else '❌ NO'}")
        print("="*70)
    
    return {
        'success': success,
        'steps': step,
        'reward': total_reward,
        'final_metric': final_metric,
        'target_metric': target_metric
    }


def main():
    """CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Run ReproAgent inference"
    )
    
    parser.add_argument(
        '--difficulty',
        type=str,
        default='easy',
        choices=['easy', 'medium', 'hard'],
        help='Difficulty level'
    )
    
    parser.add_argument(
        '--agent',
        type=str,
        default='reasoning',
        choices=['reasoning', 'random', 'rl'],
        help='Agent type'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=30,
        help='Maximum steps'
    )
    
    parser.add_argument(
        '--llm',
        action='store_true',
        help='Enable LLM (requires API key)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Number of episodes to run'
    )
    
    args = parser.parse_args()
    
    if args.episodes == 1:
        # Single episode
        result = run_inference(
            difficulty=args.difficulty,
            agent_type=args.agent,
            max_steps=args.steps,
            use_llm=args.llm,
            verbose=not args.quiet
        )
        
        sys.exit(0 if result['success'] else 1)
    
    else:
        # Multiple episodes
        print(f"\n🔄 Running {args.episodes} episodes...\n")
        
        results = []
        for i in range(args.episodes):
            print(f"\nEpisode {i+1}/{args.episodes}")
            print("-"*70)
            
            result = run_inference(
                difficulty=args.difficulty,
                agent_type=args.agent,
                max_steps=args.steps,
                use_llm=args.llm,
                verbose=False
            )
            
            results.append(result)
            
            print(f"Success: {result['success']} | Metric: {result['final_metric']:.3f}")
        
        # Summary
        success_rate = sum(r['success'] for r in results) / len(results)
        avg_metric = sum(r['final_metric'] for r in results) / len(results)
        avg_steps = sum(r['steps'] for r in results) / len(results)
        
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        print(f"Success Rate: {success_rate*100:.1f}%")
        print(f"Avg Metric: {avg_metric:.3f}")
        print(f"Avg Steps: {avg_steps:.1f}")
        print("="*70)


if __name__ == "__main__":
    main()
