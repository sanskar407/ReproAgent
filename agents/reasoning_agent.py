"""
Main reasoning agent - orchestrates the entire reproduction workflow.
Uses hypothesis-driven approach to intelligently navigate the reproduction process.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from reproagent.environment import ReproAgentEnv
from reproagent.state import ReproductionState, Phase
from reproagent.actions import ActionSpace, ActionType, Action
from reproagent.models import LLMClient
from agents.paper_parser import PaperParser
from agents.repo_analyzer import RepoAnalyzer
from agents.debugger import Debugger


class ReasoningAgent:
    """
    Main intelligent agent for paper reproduction.
    
    Strategy:
    1. Parse paper → understand what to reproduce
    2. Find & analyze repo → understand how to reproduce
    3. Setup environment → prepare for execution
    4. Execute & debug → run code, fix errors
    5. Experiment → tune hyperparameters
    6. Compare → validate reproduction
    """
    
    def __init__(self, env: ReproAgentEnv, use_llm: bool = True):
        """
        Args:
            env: ReproAgent environment
            use_llm: Whether to use LLM for reasoning
        """
        self.env = env
        self.action_space = ActionSpace()
        self.use_llm = use_llm
        
        # Initialize LLM and sub-agents
        if use_llm:
            try:
                self.llm = LLMClient()
            except:
                print("⚠️  LLM not available, using rule-based mode")
                self.llm = LLMClient(provider="mock")
                self.use_llm = False
        else:
            self.llm = LLMClient(provider="mock")
        
        self.paper_parser = PaperParser(self.llm)
        self.repo_analyzer = RepoAnalyzer(self.llm)
        self.debugger = Debugger(self.llm)
        
        # Agent state
        self.current_strategy = "systematic"  # systematic, debugging, experimenting
        self.hypotheses = []
        self.phase_progress = {
            Phase.PARSING: False,
            Phase.REPO_ANALYSIS: False,
            Phase.SETUP: False,
            Phase.EXECUTION: False,
            Phase.DEBUGGING: False,
            Phase.EXPERIMENTATION: False,
        }
    
    def select_action(
        self, 
        observation: Dict[str, np.ndarray], 
        info: Dict[str, Any]
    ) -> int:
        """
        Select next action based on current state.
        
        Args:
            observation: Environment observation
            info: Additional info
            
        Returns:
            Action ID
        """
        # Get current state from environment
        state = self.env.state
        
        # Determine strategy based on phase
        if state.meta.phase == Phase.IDLE or state.meta.phase == Phase.PARSING:
            return self._parsing_phase_action(state)
        
        elif state.meta.phase == Phase.REPO_ANALYSIS:
            return self._repo_analysis_action(state)
        
        elif state.meta.phase == Phase.SETUP:
            return self._setup_phase_action(state)
        
        elif state.meta.phase == Phase.EXECUTION:
            return self._execution_phase_action(state)
        
        elif state.meta.phase == Phase.DEBUGGING:
            return self._debugging_phase_action(state)
        
        elif state.meta.phase == Phase.EXPERIMENTATION:
            return self._experimentation_action(state)
        
        else:
            # Default: random exploration
            return self.env.action_space.sample()
    
    def _parsing_phase_action(self, state: ReproductionState) -> int:
        """Actions for paper parsing phase."""
        
        if not state.paper.parsed:
            return self.action_space.get_id_by_action(ActionType.PARSE_PDF)
        
        elif not state.paper.github_links:
            return self.action_space.get_id_by_action(ActionType.EXTRACT_GITHUB)
        
        elif state.paper.target_metric == 0.0:
            return self.action_space.get_id_by_action(ActionType.EXTRACT_METRICS)
        
        else:
            # Parsing is complete — move to repo cloning
            if not state.repo.cloned:
                return self.action_space.get_id_by_action(ActionType.CLONE_REPO)
            else:
                return self.action_space.get_id_by_action(ActionType.READ_README)
    
    def _repo_analysis_action(self, state: ReproductionState) -> int:
        """Actions for repository analysis phase."""
        
        if not state.repo.cloned and state.paper.github_links:
            return self.action_space.get_id_by_action(ActionType.CLONE_REPO)
        
        elif state.repo.cloned and not state.repo.readme_content:
            return self.action_space.get_id_by_action(ActionType.READ_README)
        
        elif state.repo.readme_content and not state.repo.entry_point:
            return self.action_space.get_id_by_action(ActionType.FIND_ENTRY_POINT)
        
        elif state.repo.entry_point and not state.repo.dependencies:
            return self.action_space.get_id_by_action(ActionType.EXTRACT_DEPS)
        
        else:
            # Repo fully analyzed — move to environment setup (CREATE_VENV first!)
            return self.action_space.get_id_by_action(ActionType.CREATE_VENV)
    
    def _setup_phase_action(self, state: ReproductionState) -> int:
        """Actions for environment setup phase."""
        
        if not state.environment.setup_complete:
            if state.repo.dependencies:
                return self.action_space.get_id_by_action(ActionType.INSTALL_REQUIREMENTS)
            else:
                # Even with no explicit deps listed, verify setup
                return self.action_space.get_id_by_action(ActionType.VERIFY_SETUP)
        
        else:
            # Setup complete — move to execution
            return self.action_space.get_id_by_action(ActionType.RUN_TRAINING)
    
    def _execution_phase_action(self, state: ReproductionState) -> int:
        """Actions for code execution phase."""
        
        if state.execution.last_error:
            # Transition to debugging
            return self.action_space.get_id_by_action(ActionType.ANALYZE_ERROR)
        
        elif state.experiment.current_metric > 0 and state.experiment.gap > 0.05:
            # Has some results but gap is large — move to experimentation
            return self.action_space.get_id_by_action(ActionType.RUN_EXPERIMENT)
        
        elif state.experiment.current_metric > 0 and state.experiment.gap <= 0.05:
            # Close enough — compare
            return self.action_space.get_id_by_action(ActionType.COMPARE_RESULTS)
        
        else:
            # Run training
            return self.action_space.get_id_by_action(ActionType.RUN_TRAINING)
    
    def _debugging_phase_action(self, state: ReproductionState) -> int:
        """Actions for debugging phase."""
        
        if state.debug.current_error and not state.debug.last_hypothesis:
            return self.action_space.get_id_by_action(ActionType.ANALYZE_ERROR)
        
        elif state.debug.last_hypothesis and len(state.debug.fix_attempts) == 0:
            return self.action_space.get_id_by_action(ActionType.SEARCH_SOLUTION)
        
        elif state.debug.current_error and len(state.debug.solutions_tried) < 3:
            return self.action_space.get_id_by_action(ActionType.APPLY_FIX)
        
        elif state.debug.current_error:
            return self.action_space.get_id_by_action(ActionType.APPLY_FIX)
        
        else:
            # Error resolved — back to training
            return self.action_space.get_id_by_action(ActionType.RUN_TRAINING)
    
    def _experimentation_action(self, state: ReproductionState) -> int:
        """Actions for hyperparameter tuning phase."""
        
        gap = state.experiment.gap
        experiments_run = state.experiment.experiments_run
        
        # Use LLM for intelligent hyperparameter selection if available
        if self.use_llm and experiments_run > 0:
            action = self._llm_suggest_hyperparameter_action(state)
            if action is not None:
                return action
        
        # Rule-based: alternate between tuning a param and running an experiment
        if experiments_run > 0 and experiments_run % 2 == 0:
            # Every other step, run an experiment to measure progress
            return self.action_space.get_id_by_action(ActionType.RUN_EXPERIMENT)
        
        if gap > 0.3:
            return self.action_space.get_id_by_action(ActionType.MODIFY_LR)
        elif gap > 0.15:
            if experiments_run % 4 < 2:
                return self.action_space.get_id_by_action(ActionType.MODIFY_BATCH)
            else:
                return self.action_space.get_id_by_action(ActionType.MODIFY_OPTIMIZER)
        elif gap > 0.05:
            return self.action_space.get_id_by_action(ActionType.ADD_REGULARIZATION)
        else:
            # Very close — run experiment to lock in
            return self.action_space.get_id_by_action(ActionType.RUN_EXPERIMENT)
    
    def _llm_suggest_hyperparameter_action(self, state: ReproductionState) -> Optional[int]:
        """Use LLM to suggest next hyperparameter action."""
        
        prompt = f"""
You are tuning hyperparameters to reproduce a paper's results.

Current state:
- Target metric: {state.paper.target_metric:.3f}
- Current metric: {state.experiment.current_metric:.3f}
- Gap: {state.experiment.gap:.3f}
- Experiments run: {state.experiment.experiments_run}
- Current config: {state.experiment.current_config}

What should be adjusted next?

Options:
1. learning_rate
2. batch_size
3. optimizer
4. epochs
5. regularization
6. run_experiment (test current config)

Respond with JSON:
{{
    "action": "learning_rate",
    "reasoning": "why this action"
}}
"""
        
        try:
            result = self.llm.generate_structured(prompt)
            action_name = result.get('action', '')
            
            action_map = {
                'learning_rate': ActionType.MODIFY_LR,
                'batch_size': ActionType.MODIFY_BATCH,
                'optimizer': ActionType.MODIFY_OPTIMIZER,
                'epochs': ActionType.MODIFY_EPOCHS,
                'regularization': ActionType.ADD_REGULARIZATION,
                'run_experiment': ActionType.RUN_EXPERIMENT
            }
            
            if action_name in action_map:
                action_type = action_map[action_name]
                return self.action_space.get_id_by_action(action_type)
        
        except Exception as e:
            print(f"⚠️  LLM suggestion failed: {e}")
        
        return None
    
    def form_hypothesis(self, state: ReproductionState) -> str:
        """
        Form hypothesis about what's preventing reproduction.
        
        Args:
            state: Current state
            
        Returns:
            Hypothesis string
        """
        if not state.paper.parsed:
            return "Need to parse paper to understand target"
        
        elif not state.repo.cloned:
            return "Need to find and clone repository"
        
        elif state.debug.current_error:
            return f"Need to fix error: {state.debug.current_error[:50]}"
        
        elif state.experiment.gap > 0.2:
            return "Hyperparameters are significantly off from optimal"
        
        elif state.experiment.gap > 0.05:
            return "Need fine-tuning of hyperparameters"
        
        else:
            return "Close to target, validating reproduction"
    
    def get_reasoning(self, state: ReproductionState, action_id: int) -> str:
        """
        Generate human-readable reasoning for action.
        
        Args:
            state: Current state
            action_id: Selected action
            
        Returns:
            Reasoning string
        """
        action_type = self.action_space.get_action_by_id(action_id)
        
        reasoning_map = {
            ActionType.PARSE_PDF: f"📄 Parsing paper to extract methodology",
            ActionType.EXTRACT_GITHUB: f"🔍 Looking for implementation repository",
            ActionType.CLONE_REPO: f"📥 Cloning repository: {state.paper.github_links[0] if state.paper.github_links else 'unknown'}",
            ActionType.READ_README: f"📖 Reading setup instructions",
            ActionType.INSTALL_REQUIREMENTS: f"📦 Installing {len(state.repo.dependencies)} dependencies",
            ActionType.RUN_TRAINING: f"🚀 Executing training script",
            ActionType.ANALYZE_ERROR: f"🔍 Analyzing error: {state.debug.current_error[:30]}...",
            ActionType.APPLY_FIX: f"🔧 Applying fix attempt #{len(state.debug.fix_attempts) + 1}",
            ActionType.RUN_EXPERIMENT: f"🧪 Running experiment #{state.experiment.experiments_run + 1}",
            ActionType.MODIFY_LR: f"⚙️  Adjusting learning rate (gap: {state.experiment.gap:.3f})",
            ActionType.COMPARE_RESULTS: f"📊 Comparing results: {state.experiment.current_metric:.3f} vs {state.paper.target_metric:.3f}",
        }
        
        return reasoning_map.get(action_type, f"Executing {action_type.value}")
    
    def reset(self):
        """Reset agent for new episode."""
        self.current_strategy = "systematic"
        self.hypotheses = []
        self.phase_progress = {phase: False for phase in Phase}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'strategy': self.current_strategy,
            'hypotheses_formed': len(self.hypotheses),
            'phases_completed': sum(self.phase_progress.values())
        }


class RLAgent:
    """
    RL-trainable agent (for PPO/DPO training).
    Uses neural network policy.
    """
    
    def __init__(self, env: ReproAgentEnv, policy_network=None):
        """
        Args:
            env: Environment
            policy_network: Pre-trained policy (optional)
        """
        self.env = env
        self.policy = policy_network
        
        if policy_network is None:
            self._init_policy()
    
    def _init_policy(self):
        """Initialize policy network."""
        try:
            import torch
            import torch.nn as nn
            
            # Simple MLP policy
            obs_dim = 25  # 5 feature vectors × 5 dims each
            action_dim = self.env.action_space.n
            
            self.policy = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            )
        except ImportError:
            print("⚠️  PyTorch not installed, using random policy")
            self.policy = None
    
    def select_action(
        self, 
        observation: Dict[str, np.ndarray], 
        info: Dict[str, Any]
    ) -> int:
        """Select action using policy network."""
        
        if self.policy is None:
            return self.env.action_space.sample()
        
        try:
            import torch
            
            # Flatten observation
            obs_vec = np.concatenate([
                observation['paper_features'],
                observation['repo_features'],
                observation['execution_features'],
                observation['experiment_features'],
                observation['meta_features']
            ])
            
            obs_tensor = torch.FloatTensor(obs_vec).unsqueeze(0)
            
            with torch.no_grad():
                action_probs = self.policy(obs_tensor)
            
            # Sample action
            action = torch.multinomial(action_probs, 1).item()
            
            return action
        except:
            return self.env.action_space.sample()
    
    def reset(self):
        """Reset agent."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats."""
        return {'type': 'RL'}


# Factory function
def create_agent(env: ReproAgentEnv, agent_type: str = "reasoning", **kwargs):
    """
    Factory function to create agents.
    
    Args:
        env: Environment
        agent_type: 'reasoning', 'rl', or 'random'
        **kwargs: Additional arguments
        
    Returns:
        Agent instance
    """
    if agent_type == "reasoning":
        return ReasoningAgent(env, use_llm=kwargs.get('use_llm', True))
    
    elif agent_type == "rl":
        return RLAgent(env, policy_network=kwargs.get('policy', None))
    
    elif agent_type == "random":
        # Simple random agent for baseline
        class RandomAgent:
            def __init__(self, env):
                self.env = env
            
            def select_action(self, obs, info):
                return self.env.action_space.sample()
            
            def reset(self):
                pass
            
            def get_stats(self):
                return {'type': 'random'}
            
            def get_reasoning(self, state, action_id):
                return f"Random action: {action_id}"
        
        return RandomAgent(env)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# Test
if __name__ == "__main__":
    from reproagent.environment import ReproAgentEnv
    
    # Create environment
    env = ReproAgentEnv(difficulty="easy", use_llm=False)
    
    # Create agent
    agent = create_agent(env, agent_type="reasoning", use_llm=False)
    
    # Run episode
    obs, info = env.reset()
    
    for step in range(20):
        action = agent.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step + 1}: {info.get('action_type', 'unknown')} | Reward: {reward:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\nFinal metric: {info.get('current_metric', 0.0):.3f}")
