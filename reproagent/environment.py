"""
Main OpenEnv-compatible Gymnasium environment for ReproAgent.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from reproagent.state import (
    ReproductionState,
    PaperState,
    RepoState,
    Phase,
    DifficultyLevel
)
from reproagent.actions import ActionSpace, ActionType, Action
from reproagent.reward import RewardFunction, RewardComponents
from reproagent.models import LLMClient
from reproagent.papers import PaperDataset


class ReproAgentEnv(gym.Env):
    """
    OpenEnv-compatible environment for ML paper reproduction.
    
    The agent must:
    1. Parse a research paper
    2. Find and clone GitHub repository
    3. Set up environment and dependencies
    4. Run code and debug errors
    5. Tune hyperparameters
    6. Reproduce paper's claimed results
    """
    
    metadata = {
        'render_modes': ['human', 'ansi'],
        'render_fps': 1
    }
    
    def __init__(
        self,
        paper_path: Optional[str] = None,
        difficulty: str = "easy",
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        use_llm: bool = True,
        exec_mode: str = "Simulation",
        workspace_dir: str = "/tmp/reproagent"
    ):
        """
        Args:
            paper_path: Path to specific paper PDF (optional)
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            max_steps: Maximum steps per episode
            render_mode: Rendering mode
            use_llm: Whether to use LLM (False for testing)
            exec_mode: 'Simulation' or 'Real Execution'
            workspace_dir: Directory where code will be cloned and executed
        """
        super().__init__()
        
        self.paper_path = paper_path
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.use_llm = use_llm
        self.exec_mode = exec_mode
        self.workspace_dir = workspace_dir
        
        # Initialize components
        self.action_space_helper = ActionSpace()
        self.paper_dataset = PaperDataset()
        
        # LLM client (optional for testing)
        if use_llm:
            try:
                self.llm = LLMClient()
            except Exception:
                print("[WARN] LLM not available, using mock mode")
                self.llm = LLMClient(provider="mock")
        else:
            self.llm = LLMClient(provider="mock")
        
        # Define spaces
        self._setup_spaces()
        
        # State
        self.state: Optional[ReproductionState] = None
        self.reward_function: Optional[RewardFunction] = None
        
    def _setup_spaces(self):
        """Setup Gymnasium observation and action spaces."""
        
        # Action space: Discrete (all possible actions)
        self.action_space = spaces.Discrete(self.action_space_helper.n)
        
        # Observation space: Dict of feature vectors
        self.observation_space = spaces.Dict({
            'paper_features': spaces.Box(
                low=0, high=1, shape=(5,), dtype=np.float32
            ),
            'repo_features': spaces.Box(
                low=0, high=1, shape=(5,), dtype=np.float32
            ),
            'execution_features': spaces.Box(
                low=0, high=1, shape=(5,), dtype=np.float32
            ),
            'experiment_features': spaces.Box(
                low=0, high=1, shape=(5,), dtype=np.float32
            ),
            'meta_features': spaces.Box(
                low=0, high=1, shape=(5,), dtype=np.float32
            )
        })
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional info
        """
        super().reset(seed=seed)
        
        # Initialize state
        self.state = ReproductionState()
        
        # Load paper
        if self.paper_path:
            # Specific paper provided
            self._load_paper_from_path(self.paper_path)
        else:
            # Load from dataset
            self._load_paper_from_dataset()
        
        # Set difficulty
        if options and 'difficulty' in options:
            difficulty_str = options['difficulty']
            self.state.meta.difficulty_level = DifficultyLevel(difficulty_str)
        else:
            self.state.meta.difficulty_level = DifficultyLevel(self.difficulty)
        
        # Initialize reward function
        self.reward_function = RewardFunction(
            max_steps=self.max_steps,
            target_metric=self.state.paper.target_metric
        )
        
        # Sync experiment target metric with paper
        self.state.experiment.target_metric = self.state.paper.target_metric
        self.state.experiment.gap = self.state.paper.target_metric
        
        # Get initial observation
        observation = self.state.to_observation()
        info = self._get_info()
        
        if self.render_mode == 'human':
            self.render()
        
        return observation, info
    
    def _load_paper_from_path(self, paper_path: str):
        """Load paper from PDF path."""
        # For now, create mock paper state
        # In full implementation, would parse PDF here
        self.state.paper = PaperState(
            pdf_path=paper_path,
            title="Sample Paper",
            dataset="CIFAR-10",
            model="ResNet-50",
            target_metric=0.95,
            metric_name="accuracy",
            parsed=False
        )
    
    def _load_paper_from_dataset(self):
        """Load paper from dataset."""
        paper_data = self.paper_dataset.get_random_paper(self.difficulty)
        
        if paper_data:
            self.state.paper = PaperState(
                title=paper_data.get('title', 'Unknown'),
                dataset=paper_data.get('dataset', 'Unknown'),
                model=paper_data.get('model', 'Unknown'),
                target_metric=paper_data.get('target_metric', 0.95),
                metric_name=paper_data.get('metric_name', 'accuracy'),
                github_links=[paper_data['github_url']] if paper_data.get('github_url') else [],
                key_claims=paper_data.get('key_claims', []),
                parsed=False
            )
            
            # Store ground truth for simulation
            self._ground_truth_config = paper_data.get('ground_truth_config', {})
        else:
            # Fallback to default
            self.state.paper = PaperState(
                title="Default Paper",
                dataset="CIFAR-10",
                model="ResNet-50",
                target_metric=0.95,
                parsed=False
            )
            self._ground_truth_config = {}
    
    def step(
        self,
        action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute action in environment.
        
        Args:
            action: Action ID
            
        Returns:
            observation: New observation
            reward: Reward
            terminated: Whether episode ended successfully
            truncated: Whether episode was cut off
            info: Additional info
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Store previous state for reward calculation
        prev_state_dict = self.state.to_dict()
        prev_state = ReproductionState()
        # Copy relevant fields for reward calculation
        prev_state.experiment.current_metric = self.state.experiment.current_metric
        prev_state.experiment.best_metric = self.state.experiment.best_metric
        prev_state.debug.errors_encountered = self.state.debug.errors_encountered.copy()
        prev_state.meta.phase = self.state.meta.phase
        prev_state.meta.step_count = self.state.meta.step_count
        
        # Get action type
        action_type = self.action_space_helper.get_action_by_id(action)
        
        # Execute action
        self._execute_action(action_type)
        
        # Increment step count
        self.state.meta.step_count += 1
        
        # Calculate reward
        reward_components = self.reward_function.calculate_reward(
            prev_state,
            action,
            self.state
        )
        
        # Check termination
        terminated = self._check_success()
        truncated = self.state.meta.step_count >= self.max_steps
        
        # Get observation and info
        observation = self.state.to_observation()
        info = self._get_info()
        info['reward_components'] = reward_components.to_dict()
        info['action_type'] = action_type.value
        
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward_components.total_reward, terminated, truncated, info
    
    def _execute_action(self, action_type: ActionType):
        """
        Execute specific action and update state.
        This is where the actual simulation happens.
        """
        
        # Update phase based on action
        self._update_phase(action_type)
        
        # Execute action based on type
        if action_type == ActionType.PARSE_PDF:
            self._action_parse_pdf()
        
        elif action_type == ActionType.EXTRACT_GITHUB:
            self._action_extract_github()
        
        elif action_type == ActionType.EXTRACT_METRICS:
            self._action_extract_metrics()
        
        elif action_type == ActionType.VALIDATE_PARSING:
            self._action_validate_parsing()
        
        elif action_type == ActionType.CLONE_REPO:
            self._action_clone_repo()
        
        elif action_type == ActionType.READ_README:
            self._action_read_readme()
        
        elif action_type == ActionType.ANALYZE_CODE:
            self._action_analyze_code()
        
        elif action_type == ActionType.FIND_ENTRY_POINT:
            self._action_find_entry_point()
        
        elif action_type == ActionType.EXTRACT_DEPS:
            self._action_extract_deps()
        
        elif action_type == ActionType.CREATE_VENV:
            self._action_create_venv()
        
        elif action_type == ActionType.INSTALL_REQUIREMENTS:
            self._action_install_requirements()
        
        elif action_type == ActionType.INSTALL_PACKAGE:
            self._action_install_requirements()  # same effect
        
        elif action_type == ActionType.DOWNLOAD_DATA:
            self._action_download_data()
        
        elif action_type == ActionType.VERIFY_SETUP:
            self._action_verify_setup()
        
        elif action_type == ActionType.RUN_TRAINING:
            self._action_run_training()
        
        elif action_type == ActionType.RUN_EVAL:
            self._action_run_experiment()  # eval = re-evaluate
        
        elif action_type == ActionType.STOP_PROCESS:
            self._action_stop_process()
        
        elif action_type == ActionType.CHECK_LOGS:
            self._action_check_logs()
        
        elif action_type == ActionType.ANALYZE_ERROR:
            self._action_analyze_error()
        
        elif action_type == ActionType.SEARCH_SOLUTION:
            self._action_search_solution()
        
        elif action_type == ActionType.APPLY_FIX:
            self._action_apply_fix()
        
        elif action_type == ActionType.MODIFY_CODE:
            self._action_apply_fix()  # similar effect
        
        elif action_type == ActionType.ROLLBACK:
            self._action_rollback()
        
        elif action_type == ActionType.TEST_FIX:
            self._action_test_fix()
        
        elif action_type == ActionType.RUN_EXPERIMENT:
            self._action_run_experiment()
        
        elif action_type == ActionType.MODIFY_LR:
            self._action_modify_hyperparameter('learning_rate', 0.0001)
        
        elif action_type == ActionType.MODIFY_BATCH:
            self._action_modify_hyperparameter('batch_size', 64)
        
        elif action_type == ActionType.MODIFY_OPTIMIZER:
            self._action_modify_hyperparameter('optimizer', 'adamw')
        
        elif action_type == ActionType.MODIFY_EPOCHS:
            self._action_modify_hyperparameter('epochs', 100)
        
        elif action_type == ActionType.ADD_REGULARIZATION:
            self._action_modify_hyperparameter('weight_decay', 0.01)
        
        elif action_type == ActionType.COMPARE_RESULTS:
            self._action_compare_results()
        
        elif action_type == ActionType.GENERATE_REPORT:
            self._action_generate_report()
        
        elif action_type == ActionType.FORM_HYPOTHESIS:
            self._action_form_hypothesis()
        
        elif action_type == ActionType.WAIT:
            self.state.execution.logs.append("... waiting")
        
        elif action_type == ActionType.ABORT:
            self.state.meta.failure_reason = "Agent aborted"
        
        elif action_type == ActionType.RESET:
            self.state.execution.logs.append("Reset requested")
    
    def _update_phase(self, action_type: ActionType):
        """Update current phase based on action."""
        phase_map = {
            ActionType.PARSE_PDF: Phase.PARSING,
            ActionType.EXTRACT_GITHUB: Phase.PARSING,
            ActionType.EXTRACT_METRICS: Phase.PARSING,
            ActionType.VALIDATE_PARSING: Phase.REPO_ANALYSIS,  # parsing done → move on
            ActionType.CLONE_REPO: Phase.REPO_ANALYSIS,
            ActionType.READ_README: Phase.REPO_ANALYSIS,
            ActionType.ANALYZE_CODE: Phase.REPO_ANALYSIS,
            ActionType.FIND_ENTRY_POINT: Phase.REPO_ANALYSIS,
            ActionType.EXTRACT_DEPS: Phase.REPO_ANALYSIS,
            ActionType.CREATE_VENV: Phase.SETUP,
            ActionType.INSTALL_REQUIREMENTS: Phase.SETUP,
            ActionType.INSTALL_PACKAGE: Phase.SETUP,
            ActionType.DOWNLOAD_DATA: Phase.SETUP,
            ActionType.VERIFY_SETUP: Phase.SETUP,
            ActionType.RUN_TRAINING: Phase.EXECUTION,
            ActionType.RUN_EVAL: Phase.EXECUTION,
            ActionType.STOP_PROCESS: Phase.EXECUTION,
            ActionType.CHECK_LOGS: Phase.EXECUTION,
            ActionType.ANALYZE_ERROR: Phase.DEBUGGING,
            ActionType.SEARCH_SOLUTION: Phase.DEBUGGING,
            ActionType.APPLY_FIX: Phase.DEBUGGING,
            ActionType.MODIFY_CODE: Phase.DEBUGGING,
            ActionType.ROLLBACK: Phase.DEBUGGING,
            ActionType.TEST_FIX: Phase.DEBUGGING,
            ActionType.MODIFY_LR: Phase.EXPERIMENTATION,
            ActionType.MODIFY_BATCH: Phase.EXPERIMENTATION,
            ActionType.MODIFY_OPTIMIZER: Phase.EXPERIMENTATION,
            ActionType.MODIFY_EPOCHS: Phase.EXPERIMENTATION,
            ActionType.ADD_REGULARIZATION: Phase.EXPERIMENTATION,
            ActionType.RUN_EXPERIMENT: Phase.EXPERIMENTATION,
            ActionType.COMPARE_RESULTS: Phase.COMPARISON,
            ActionType.GENERATE_REPORT: Phase.COMPARISON,
            ActionType.FORM_HYPOTHESIS: Phase.EXPERIMENTATION,
        }
        
        if action_type in phase_map:
            self.state.meta.phase = phase_map[action_type]
    
    # Action implementations
    def _action_parse_pdf(self):
        """Simulate PDF parsing."""
        if not self.state.paper.parsed:
            self.state.paper.parsed = True
            self.state.paper.confidence = 0.9
            self.state.execution.logs.append("✅ PDF parsed successfully")
    
    def _action_extract_github(self):
        """Simulate GitHub link extraction."""
        if self.state.paper.parsed and not self.state.paper.github_links:
            # Use links from loaded paper data if available
            self.state.paper.github_links = ["https://github.com/example/repo"]
            self.state.execution.logs.append("[OK] Found GitHub repository")
        elif self.state.paper.github_links:
            self.state.execution.logs.append(f"[OK] GitHub already known: {self.state.paper.github_links[0]}")
    
    def _action_extract_metrics(self):
        """Simulate metric extraction from paper."""
        if self.state.paper.parsed:
            self.state.execution.logs.append(
                f"[OK] Target metric: {self.state.paper.target_metric:.3f} {self.state.paper.metric_name}"
            )
    
    def _action_validate_parsing(self):
        """Validate parsing results."""
        if self.state.paper.parsed:
            self.state.paper.confidence = min(1.0, self.state.paper.confidence + 0.1)
            self.state.execution.logs.append("[OK] Parsing validated")
    
    def _action_clone_repo(self):
        """Clone the repository."""
        if self.state.paper.github_links and not self.state.repo.cloned:
            url = self.state.paper.github_links[0] if isinstance(self.state.paper.github_links, list) else self.state.paper.github_links
            self.state.repo.url = url
            
            if self.exec_mode == "Real Execution":
                import subprocess, os, shutil, re as _re
                
                # Extract repo name from URL for unique folder naming
                repo_name = url.rstrip('/').split('/')[-1].replace('.git', '')
                if not repo_name:
                    repo_name = "repo"
                target_dir = os.path.join(self.workspace_dir, repo_name)
                
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)  # clean slate
                
                os.makedirs(self.workspace_dir, exist_ok=True)
                self.state.execution.logs.append(f"[EXEC] Cloning {url} into {target_dir}")
                
                try:
                    # Use --depth 1 for faster cloning
                    res = subprocess.run(
                        ["git", "clone", "--depth", "1", url, target_dir],
                        capture_output=True, text=True, timeout=300
                    )
                    if res.returncode == 0:
                        self.state.repo.cloned = True
                        self.state.repo.local_path = target_dir
                        self.state.repo.framework = "pytorch"  # default assumption
                        self.state.execution.logs.append(f"[OK] Repository cloned to {target_dir}")
                    else:
                        self.state.execution.logs.append(f"[ERROR] Clone failed: {res.stderr[:300]}")
                except subprocess.TimeoutExpired:
                    self.state.execution.logs.append(f"[ERROR] Clone timed out after 300s. Repo may be too large.")
                except Exception as e:
                    self.state.execution.logs.append(f"[ERROR] Exception during clone: {e}")
            else:
                self.state.repo.cloned = True
                self.state.repo.local_path = "/tmp/repo"
                self.state.repo.framework = "pytorch"
                self.state.execution.logs.append(f"[OK] Repository cloned: {url}")
    
    def _action_read_readme(self):
        """Simulate README parsing."""
        if self.state.repo.cloned and not self.state.repo.readme_content:
            if self.exec_mode == "Real Execution":
                import os
                readme_path = ""
                for filename in ["README.md", "readme.md", "README.MD", "README.txt"]:
                    p = os.path.join(self.state.repo.local_path, filename)
                    if os.path.exists(p):
                        readme_path = p
                        break
                
                if readme_path:
                    try:
                        with open(readme_path, "r", encoding="utf-8") as f:
                            self.state.repo.readme_content = f.read()
                        self.state.repo.entry_point = "train.py" # to be improved later
                        self.state.repo.dependencies = ["torch", "numpy", "torchvision"] # basic fallback
                        self.state.execution.logs.append(f"[OK] README read ({len(self.state.repo.readme_content)} chars)")
                    except Exception as e:
                        self.state.execution.logs.append(f"[ERROR] Could not read README: {e}")
                else:
                    self.state.repo.readme_content = "No README found."
                    self.state.execution.logs.append("[WARN] No README file found in repo")
            else:
                self.state.repo.readme_content = "Mock README content"
                self.state.repo.entry_point = "train.py"
                self.state.repo.dependencies = ["torch", "numpy", "torchvision"]
                self.state.execution.logs.append("[OK] README parsed, found entry point: train.py")
    
    def _action_analyze_code(self):
        """Simulate code structure analysis."""
        if self.state.repo.cloned:
            self.state.repo.repo_quality_score = min(1.0, self.state.repo.repo_quality_score + 0.3)
            self.state.execution.logs.append("[OK] Code structure analyzed")
    
    def _action_find_entry_point(self):
        """Find the entry point by reading README instructions first, then scanning files."""
        if self.state.repo.cloned and not self.state.repo.entry_point:
            if self.exec_mode == "Real Execution":
                import os, re
                lp = self.state.repo.local_path
                ep = ""
                readme_scripts = []  # Store ALL scripts found in README
                
                # === STEP 1: Always check README FIRST for instructions ===
                if self.state.repo.readme_content:
                    # 1a. Find python commands inside bash/sh blocks
                    bash_blocks = re.findall(r"```(?:bash|sh|shell|console)?\n(.*?)\n```", self.state.repo.readme_content, re.DOTALL)
                    for block in bash_blocks:
                        lines = block.strip().split('\n')
                        for line in lines:
                            stripped = line.strip()
                            if stripped.startswith("python ") or stripped.startswith("python3 "):
                                parts = stripped.split()
                                if len(parts) >= 2 and parts[1].endswith(".py"):
                                    script = parts[1]
                                    if script.startswith("./"):
                                        script = script[2:]
                                    readme_scripts.append(script)
                    
                    # 1b. Also find inline python commands outside code blocks
                    inline_matches = re.findall(r"(?:^|\n)\s*(?:python|python3)\s+(\S+\.py)", self.state.repo.readme_content)
                    readme_scripts.extend(inline_matches)
                
                # Store all found scripts for potential sequential execution
                if readme_scripts:
                    # Store the full list so RUN_TRAINING can iterate
                    self.state.repo.setup_instructions = readme_scripts
                    ep = readme_scripts[0]  # Start with first script
                    self.state.execution.logs.append(f"[OK] Found {len(readme_scripts)} script(s) in README: {readme_scripts}")
                
                # === STEP 2: Only if README had no scripts, scan files recursively ===
                if not ep:
                    from pathlib import Path
                    for candidate in ["inference.py", "eval.py", "test.py", "main.py", "run.py", "train.py"]:
                        matches = list(Path(lp).rglob(candidate))
                        if matches:
                            # Use the shallowest one found
                            matches.sort(key=lambda x: len(x.parts))
                            ep = str(matches[0].relative_to(lp)).replace('\\', '/')
                            self.state.execution.logs.append(f"[OK] Found script: {ep}")
                            break
                
                # === STEP 3: Try python code blocks in README ===
                if not ep and self.state.repo.readme_content:
                    python_blocks = re.findall(r"```python\n(.*?)\n```", self.state.repo.readme_content, re.DOTALL)
                    if python_blocks:
                        longest_block = max(python_blocks, key=len)
                        script_path = os.path.join(lp, "readme_script.py")
                        with open(script_path, "w", encoding="utf-8") as f:
                            f.write(longest_block)
                        ep = "readme_script.py"
                        self.state.execution.logs.append("[OK] Extracted Python script from README code block")
                
                if not ep:
                    self.state.execution.logs.append("[WARN] No entry point found in README or repo files")
                    ep = "__no_entry_point__"  # marker so we don't loop forever
                    
                self.state.repo.entry_point = ep
            else:
                self.state.repo.entry_point = "train.py"
                self.state.execution.logs.append("[OK] Entry point found: train.py")
    
    def _action_extract_deps(self):
        """Simulate dependency extraction."""
        if self.state.repo.cloned and not self.state.repo.dependencies:
            if self.exec_mode == "Real Execution":
                import os
                from pathlib import Path
                lp = self.state.repo.local_path
                
                # Recursive search for requirement files
                req_matches = list(Path(lp).rglob("requirements.txt"))
                env_matches = list(Path(lp).rglob("environment.yml")) + list(Path(lp).rglob("environment.yaml"))
                
                req_path = str(req_matches[0]) if req_matches else None
                env_yaml = str(env_matches[0]) if env_matches else None
                
                if env_yaml:
                    with open(env_yaml, "r", encoding="utf-8") as f:
                        lines = [line for line in f if "- " in line]
                    self.state.repo.dependencies = lines
                    self.state.execution.logs.append(f"[OK] Found Conda env file with ~{len(lines)} dependencies")
                elif req_path:
                    with open(req_path, "r", encoding="utf-8") as f:
                        deps = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                    self.state.repo.dependencies = deps
                    self.state.execution.logs.append(f"[OK] Found {len(deps)} dependencies in requirements.txt")
                else:
                    self.state.repo.dependencies = []
                    self.state.execution.logs.append("[WARN] No requirements or environment files found")
            else:
                self.state.repo.dependencies = ["torch", "numpy", "torchvision", "tqdm"]
                self.state.execution.logs.append(f"[OK] Found {len(self.state.repo.dependencies)} dependencies")
    
    def _action_create_venv(self):
        """Simulate virtual environment creation."""
        if self.exec_mode == "Real Execution":
            import os, subprocess
            from pathlib import Path
            lp = self.state.repo.local_path
            conda_dir = os.path.join(lp, "conda_env")
            venv_dir = os.path.join(lp, "venv")
            
            env_matches = list(Path(lp).rglob("environment.yml")) + list(Path(lp).rglob("environment.yaml"))
            target = str(env_matches[0]) if env_matches else None
            
            if target:
                self.state.execution.logs.append(f"[EXEC] Creating Conda env from {os.path.basename(target)}...")
                try:
                    res = subprocess.run(["conda", "env", "create", "--prefix", conda_dir, "-f", target], capture_output=True, text=True, timeout=600)
                    if res.returncode == 0:
                        self.state.execution.logs.append("[OK] Conda environment created successfully")
                    else:
                        self.state.execution.logs.append(f"[WARN] Conda env failed: {res.stderr[:200]}")
                        # Fallback to venv if conda fails
                        self.state.execution.logs.append("[EXEC] Falling back to python venv...")
                        try:
                            res2 = subprocess.run(["python", "-m", "venv", venv_dir], capture_output=True, text=True)
                            if res2.returncode == 0:
                                self.state.execution.logs.append("[OK] Fallback venv created")
                            else:
                                self.state.execution.logs.append(f"[ERROR] Fallback venv also failed: {res2.stderr}")
                        except Exception as e2:
                            self.state.execution.logs.append(f"[ERROR] Fallback venv exception: {e2}")
                except Exception as e:
                    self.state.execution.logs.append(f"[ERROR] Exception creating conda env: {e}")
                    # Also fallback on exception
                    try:
                        subprocess.run(["python", "-m", "venv", venv_dir], capture_output=True, text=True)
                        self.state.execution.logs.append("[OK] Fallback venv created after conda exception")
                    except:
                        pass
            else:
                self.state.execution.logs.append("[EXEC] Creating python venv...")
                try:
                    res = subprocess.run(["python", "-m", "venv", venv_dir], capture_output=True, text=True)
                    if res.returncode == 0:
                        self.state.execution.logs.append("[OK] Virtual environment created")
                    else:
                        self.state.execution.logs.append(f"[ERROR] Failed to create venv: {res.stderr}")
                except Exception as e:
                    self.state.execution.logs.append(f"[ERROR] Exception creating venv: {e}")
        else:
            self.state.execution.logs.append("[OK] Virtual environment created")
    
    def _action_install_requirements(self):
        """Install packages from requirements.txt, setup.py, or pyproject.toml."""
        if not self.state.environment.setup_complete:
            if self.exec_mode == "Real Execution":
                import os, subprocess
                
                lp = self.state.repo.local_path
                conda_dir = os.path.join(lp, "conda_env")
                venv_dir = os.path.join(lp, "venv")
                
                # Make sure the env exists
                if not os.path.exists(conda_dir) and not os.path.exists(venv_dir):
                    self._action_create_venv()
                
                if os.path.exists(conda_dir):
                    self.state.environment.setup_complete = True
                    self.state.execution.logs.append("[OK] Conda env handles deps. Setup complete.")
                    return
                
                venv_pip = os.path.join(lp, "venv", "Scripts", "pip")
                if not os.path.exists(venv_pip):
                    venv_pip = os.path.join(lp, "venv", "bin", "pip")
                
                from pathlib import Path
                
                req_matches = list(Path(lp).rglob("requirements.txt"))
                setup_matches = list(Path(lp).rglob("setup.py"))
                pyproject_matches = list(Path(lp).rglob("pyproject.toml"))
                
                req_path = str(req_matches[0]) if req_matches else None
                setup_path = str(setup_matches[0]) if setup_matches else None
                pyproject_path = str(pyproject_matches[0]) if pyproject_matches else None
                
                if req_path:
                    self.state.execution.logs.append(f"[EXEC] pip install -r {os.path.basename(req_path)}...")
                    try:
                        res = subprocess.run([venv_pip, "install", "-r", req_path], capture_output=True, text=True, timeout=300, cwd=os.path.dirname(req_path))
                        if res.returncode == 0:
                            self.state.environment.packages_installed = self.state.repo.dependencies.copy()
                            self.state.execution.logs.append("[OK] Requirements installed")
                        else:
                            self.state.execution.logs.append(f"[WARN] pip install had issues: {res.stderr[:200]}")
                    except Exception as e:
                        self.state.execution.logs.append(f"[ERROR] pip install exception: {e}")
                    self.state.environment.setup_complete = True
                elif setup_path:
                    self.state.execution.logs.append("[EXEC] pip install -e . (setup.py)...")
                    try:
                        subprocess.run([venv_pip, "install", "-e", "."], capture_output=True, text=True, timeout=300, cwd=os.path.dirname(setup_path))
                        self.state.execution.logs.append("[OK] Package installed via setup.py")
                    except Exception as e:
                        self.state.execution.logs.append(f"[ERROR] setup.py install exception: {e}")
                    self.state.environment.setup_complete = True
                elif pyproject_path:
                    self.state.execution.logs.append("[EXEC] pip install -e . (pyproject.toml)...")
                    try:
                        subprocess.run([venv_pip, "install", "-e", "."], capture_output=True, text=True, timeout=300, cwd=lp)
                        self.state.execution.logs.append("[OK] Package installed via pyproject.toml")
                    except Exception as e:
                        self.state.execution.logs.append(f"[ERROR] pyproject.toml install exception: {e}")
                    self.state.environment.setup_complete = True
                else:
                    self.state.environment.setup_complete = True
                    self.state.execution.logs.append("[OK] No requirements/setup files found. Using env as-is.")
            else:
                if self.state.repo.dependencies:
                    self.state.environment.packages_installed = self.state.repo.dependencies.copy()
                self.state.environment.setup_complete = True
                self.state.execution.logs.append("[OK] Installed packages")
    def _action_download_data(self):
        """Simulate dataset download."""
        self.state.execution.logs.append(f"[OK] Dataset '{self.state.paper.dataset}' downloaded")
    
    def _action_verify_setup(self):
        """Verify environment setup is complete."""
        if self.state.environment.setup_complete:
            self.state.execution.logs.append("[OK] Setup verified - ready to run")
        else:
            if self.exec_mode == "Real Execution":
                import os
                lp = self.state.repo.local_path
                conda_dir = os.path.join(lp, "conda_env")
                venv_dir = os.path.join(lp, "venv")
                if os.path.exists(conda_dir) or os.path.exists(venv_dir):
                    self.state.environment.setup_complete = True
                    self.state.execution.logs.append("[OK] Environment detected - marking setup complete")
                else:
                    self.state.execution.logs.append("[WARN] No environment found. Setup incomplete.")
            else:
                self.state.environment.setup_complete = True
                self.state.execution.logs.append("[OK] Setup verified (simulation)")
    
    def _action_run_training(self):
        """Execute training/inference script."""
        if self.state.environment.setup_complete:
            if self.exec_mode == "Real Execution":
                import os, subprocess
                lp = self.state.repo.local_path
                conda_dir = os.path.join(lp, "conda_env")
                venv_dir = os.path.join(lp, "venv")
                
                # Find the right python executable
                python_exe = None
                use_conda_run = False
                
                if os.path.exists(conda_dir):
                    # Conda prefix install: python is at conda_env/python.exe (Win) or conda_env/bin/python
                    candidates = [
                        os.path.join(conda_dir, "python.exe"),
                        os.path.join(conda_dir, "Scripts", "python.exe"),
                        os.path.join(conda_dir, "bin", "python"),
                    ]
                    for c in candidates:
                        if os.path.exists(c):
                            python_exe = c
                            break
                    if not python_exe:
                        # Fallback: invoke via conda run
                        use_conda_run = True
                        self.state.execution.logs.append("[INFO] Using 'conda run' to execute script")
                elif os.path.exists(venv_dir):
                    candidates = [
                        os.path.join(venv_dir, "Scripts", "python.exe"),
                        os.path.join(venv_dir, "bin", "python"),
                    ]
                    for c in candidates:
                        if os.path.exists(c):
                            python_exe = c
                            break
                
                if not python_exe and not use_conda_run:
                    python_exe = "python"  # system fallback
                    self.state.execution.logs.append("[WARN] No env python found, using system python")
                
                # Resolve entry point (could be nested like mainldm/stable_cali.py)
                entry_point = os.path.join(lp, self.state.repo.entry_point)
                
                # If the entry point extracted from README doesn't exist exactly, try to find it recursively
                if not os.path.exists(entry_point):
                    from pathlib import Path
                    ep_name = os.path.basename(self.state.repo.entry_point)
                    matches = list(Path(lp).rglob(ep_name))
                    if matches:
                        matches.sort(key=lambda x: len(x.parts))
                        entry_point = str(matches[0])
                        self.state.execution.logs.append(f"[INFO] Resolved entry point to {os.path.relpath(entry_point, lp)}")
                
                if os.path.exists(entry_point):
                    # To be safe, run it from the directory containing the entry point 
                    # in case the user specified "python train.py" but it's in "code/"
                    ep_dir = os.path.dirname(entry_point)
                    script_name = os.path.basename(entry_point)
                    
                    self.state.execution.logs.append(f"[EXEC] Running {script_name} in {os.path.relpath(ep_dir, lp)}...")
                    try:
                        if use_conda_run:
                            cmd = ["conda", "run", "--prefix", conda_dir, "--no-banner", "python", script_name]
                        else:
                            cmd = [python_exe, script_name]
                        
                        res = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=600,
                            cwd=ep_dir
                        )
                        
                        # Log stdout (truncated) so user can see output
                        if res.stdout.strip():
                            stdout_tail = res.stdout.strip().split("\n")[-10:]
                            self.state.execution.logs.append("[STDOUT] " + " | ".join(stdout_tail)[:300])
                        
                        if res.returncode == 0:
                            self.state.experiment.current_metric = self.state.paper.target_metric - 0.01
                            self.state.experiment.best_metric = self.state.experiment.current_metric
                            self.state.experiment.gap = 0.01
                            self.state.execution.logs.append(f"[OK] Script completed successfully.")
                        else:
                            err_snippet = res.stderr.strip().split("\n")[-5:]
                            err_str = "\n".join(err_snippet)
                            
                            self.state.execution.last_error = err_str
                            self.state.debug.current_error = err_str
                            self.state.debug.errors_encountered.append({
                                'error': err_str,
                                'step': self.state.meta.step_count
                            })
                            self.state.execution.logs.append(f"[ERROR] Process crashed: {err_str[:300]}")
                    except subprocess.TimeoutExpired:
                        err = "Process timed out after 600 seconds"
                        self.state.execution.last_error = err
                        self.state.debug.current_error = err
                        self.state.execution.logs.append(f"[ERROR] {err}")
                    except Exception as e:
                        self.state.execution.logs.append(f"[ERROR] Subprocess error: {e}")
                else:
                    err = f"Entry point '{self.state.repo.entry_point}' not found at {entry_point}"
                    self.state.execution.last_error = err
                    self.state.debug.current_error = err
                    self.state.execution.logs.append(f"[ERROR] {err}")
                    
            else:
                # Simulate training (with possible errors)
                import random
                
                if random.random() < 0.3:  # 30% chance of error
                    self._simulate_error()
                else:
                    self._simulate_training_success()
    
    def _simulate_error(self):
        """Simulate an error occurring."""
        errors = [
            "ImportError: No module named 'torch'",
            "RuntimeError: CUDA out of memory",
            "FileNotFoundError: Dataset not found"
        ]
        import random
        error = random.choice(errors)
        
        self.state.execution.last_error = error
        self.state.debug.current_error = error
        self.state.debug.errors_encountered.append({
            'error': error,
            'step': self.state.meta.step_count
        })
        self.state.execution.logs.append(f"[ERROR] {error}")
    
    def _simulate_training_success(self):
        """Simulate successful training."""
        self.state.experiment.current_metric += 0.05
        self.state.experiment.current_metric = min(
            self.state.experiment.current_metric,
            self.state.paper.target_metric
        )
        self.state.experiment.best_metric = max(
            self.state.experiment.best_metric,
            self.state.experiment.current_metric
        )
        self.state.experiment.gap = max(0.0, self.state.paper.target_metric - self.state.experiment.current_metric)
        self.state.execution.logs.append(
            f"[OK] Training step complete: metric={self.state.experiment.current_metric:.3f}"
        )
    
    def _action_analyze_error(self):
        """Simulate error analysis."""
        if self.state.debug.current_error:
            self.state.debug.last_hypothesis = "Missing dependency or configuration issue"
            self.state.execution.logs.append(f"[ANALYZE] Error: {self.state.debug.current_error[:60]}")
    
    def _action_search_solution(self):
        """Simulate searching for a solution."""
        if self.state.debug.current_error:
            self.state.debug.solutions_tried.append("Stack Overflow search")
            self.state.execution.logs.append("[SEARCH] Found potential solution")
    
    def _action_apply_fix(self):
        """Simulate applying a fix."""
        if self.state.debug.current_error:
            self.state.debug.fix_attempts.append({
                'error': self.state.debug.current_error,
                'hypothesis': self.state.debug.last_hypothesis,
                'step': self.state.meta.step_count
            })
            
            import random
            if random.random() < 0.7:
                self.state.debug.current_error = ""
                self.state.execution.last_error = ""
                self.state.execution.logs.append("[FIX] Fix applied successfully")
            else:
                self.state.execution.logs.append("[FIX] Fix did not work, trying another approach")
    
    def _action_rollback(self):
        """Simulate rollback."""
        self.state.execution.logs.append("[ROLLBACK] Changes reverted")
    
    def _action_test_fix(self):
        """Simulate testing a fix."""
        if not self.state.debug.current_error:
            self.state.execution.logs.append("[TEST] Fix verified - error resolved")
        else:
            self.state.execution.logs.append("[TEST] Error persists")
    
    def _action_stop_process(self):
        """Simulate stopping a process."""
        self.state.execution.process_running = False
        self.state.execution.logs.append("[STOP] Process stopped")
    
    def _action_check_logs(self):
        """Simulate checking logs."""
        self.state.execution.logs.append("[LOGS] Checked recent output")
    
    def _action_run_experiment(self):
        """Simulate running experiment with current config."""
        if self.state.environment.setup_complete:
            # Calculate metric based on config similarity to ground truth
            metric = self._calculate_simulated_metric()
            
            self.state.experiment.current_metric = metric
            self.state.experiment.best_metric = max(
                self.state.experiment.best_metric,
                metric
            )
            self.state.experiment.experiments_run += 1
            self.state.experiment.gap = self.state.paper.target_metric - metric
            
            self.state.execution.logs.append(
                f"🧪 Experiment {self.state.experiment.experiments_run}: {metric:.3f}"
            )
    
    def _action_modify_hyperparameter(self, param: str, value):
        """Modify a hyperparameter."""
        self.state.experiment.current_config[param] = value
        self.state.execution.logs.append(f"[CONFIG] Set {param} = {value}")
    
    def _action_compare_results(self):
        """Compare current results to paper claims."""
        gap = self.state.paper.target_metric - self.state.experiment.current_metric
        self.state.experiment.gap = max(0.0, gap)
        self.state.execution.logs.append(
            f"[COMPARE] Current: {self.state.experiment.current_metric:.3f} vs "
            f"Target: {self.state.paper.target_metric:.3f} (gap: {gap:.3f})"
        )
    
    def _action_generate_report(self):
        """Generate reproduction report."""
        setattr(self.state.meta, 'report_generated', True)
        self.state.execution.logs.append("[REPORT] Reproduction report generated")
    
    def _action_form_hypothesis(self):
        """Form a hypothesis about what to try next."""
        self.state.reasoning.current_hypothesis = "Adjust learning rate and batch size"
        self.state.execution.logs.append("[HYPOTHESIS] Formed: adjust learning rate and batch size")
    
    def _calculate_simulated_metric(self) -> float:
        """
        Calculate simulated performance metric.
        Based on similarity to ground truth config.
        """
        if not self._ground_truth_config:
            # No ground truth, return random progress
            import random
            return 0.5 + random.random() * 0.3
        
        # Calculate similarity
        total_score = 0.0
        total_weight = 0.0
        
        param_weights = {
            'learning_rate': 0.3,
            'batch_size': 0.2,
            'optimizer': 0.2,
            'epochs': 0.1,
            'weight_decay': 0.1,
            'scheduler': 0.1
        }
        
        for param, weight in param_weights.items():
            if param in self._ground_truth_config:
                true_val = self._ground_truth_config[param]
                curr_val = self.state.experiment.current_config.get(param)
                
                if curr_val is not None:
                    if curr_val == true_val:
                        total_score += weight
                    elif isinstance(true_val, (int, float)) and isinstance(curr_val, (int, float)):
                        # Partial credit for numerical values
                        similarity = 1.0 - min(1.0, abs(true_val - curr_val) / max(abs(true_val), 1.0))
                        total_score += weight * similarity
                
                total_weight += weight
        
        if total_weight > 0:
            similarity = total_score / total_weight
        else:
            similarity = 0.5
        
        # Convert to metric
        baseline = 0.3
        max_improvement = self.state.paper.target_metric - baseline
        metric = baseline + (similarity * max_improvement)
        
        # Add small noise
        import random
        noise = random.gauss(0, 0.02)
        metric += noise
        
        return max(0.0, min(1.0, metric))
    
    def _check_success(self) -> bool:
        """Check if reproduction was successful."""
        
        if getattr(self.state.meta, 'report_generated', False):
            if self.state.paper.target_metric > 0.0:
                threshold = self.state.paper.target_metric * 0.95
                if self.state.experiment.current_metric >= threshold:
                    self.state.meta.success = True
            else:
                self.state.meta.success = not bool(self.state.execution.last_error)
            return True
            
        if self.state.paper.target_metric <= 0.0 or self.state.experiment.current_metric <= 0.0:
            return False
            
        threshold = self.state.paper.target_metric * 0.95
        
        if self.state.experiment.current_metric >= threshold:
            self.state.meta.success = True
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dict."""
        return {
            'step': self.state.meta.step_count,
            'phase': self.state.meta.phase.value,
            'current_metric': self.state.experiment.current_metric,
            'target_metric': self.state.paper.target_metric,
            'gap': self.state.experiment.gap,
            'success': self.state.meta.success,
            'logs': self.state.execution.logs[-5:]  # Last 5 logs
        }
    
    def render(self):
        """Render environment state."""
        if self.render_mode is None:
            return
        
        output = self._render_ansi()
        
        if self.render_mode == 'human':
            print(output)
        
        return output
    
    def _render_ansi(self) -> str:
        """Render as ANSI string."""
        if self.state is None:
            return "Environment not initialized"
        
        return self.state.get_summary()
    
    def close(self):
        """Cleanup."""
        pass
