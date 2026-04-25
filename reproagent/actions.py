"""
Action space for ReproAgent.
Defines all possible actions the agent can take.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum


class ActionType(Enum):
    """All possible action types."""
    
    # Phase 1: Paper Parsing
    PARSE_PDF = "parse_pdf"
    EXTRACT_GITHUB = "extract_github"
    EXTRACT_METRICS = "extract_metrics"
    VALIDATE_PARSING = "validate_parsing"
    
    # Phase 2: Repo Analysis
    CLONE_REPO = "clone_repo"
    READ_README = "read_readme"
    ANALYZE_CODE = "analyze_code"
    FIND_ENTRY_POINT = "find_entry_point"
    EXTRACT_DEPS = "extract_dependencies"
    
    # Phase 3: Environment Setup
    CREATE_VENV = "create_virtual_env"
    INSTALL_REQUIREMENTS = "install_requirements"
    INSTALL_PACKAGE = "install_package"
    DOWNLOAD_DATA = "download_dataset"
    VERIFY_SETUP = "verify_setup"
    
    # Phase 4: Execution
    RUN_TRAINING = "run_training"
    RUN_EVAL = "run_evaluation"
    STOP_PROCESS = "stop_execution"
    CHECK_LOGS = "check_logs"
    
    # Phase 5: Debugging
    ANALYZE_ERROR = "analyze_error"
    SEARCH_SOLUTION = "search_solution"
    APPLY_FIX = "apply_fix"
    MODIFY_CODE = "modify_code"
    ROLLBACK = "rollback_changes"
    TEST_FIX = "test_fix"
    
    # Phase 6: Experimentation
    MODIFY_LR = "modify_learning_rate"
    MODIFY_BATCH = "modify_batch_size"
    MODIFY_OPTIMIZER = "modify_optimizer"
    MODIFY_EPOCHS = "modify_epochs"
    ADD_REGULARIZATION = "add_regularization"
    RUN_EXPERIMENT = "run_experiment"
    
    # Phase 7: Analysis
    COMPARE_RESULTS = "compare_results"
    GENERATE_REPORT = "generate_report"
    FORM_HYPOTHESIS = "form_hypothesis"
    
    # Meta
    WAIT = "wait"
    ABORT = "abort"
    RESET = "reset"


@dataclass
class Action:
    """Single action with parameters."""
    
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_type': self.action_type.value,
            'parameters': self.parameters,
            'reasoning': self.reasoning
        }
    
    def __str__(self) -> str:
        params_str = ', '.join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.action_type.value}({params_str})"


class ActionSpace:
    """
    Defines and manages the action space.
    Maps discrete action IDs to actual actions.
    """
    
    def __init__(self):
        # Build action mapping
        self.actions = list(ActionType)
        self.action_to_id = {action: i for i, action in enumerate(self.actions)}
        self.id_to_action = {i: action for i, action in enumerate(self.actions)}
        
        # Parameterized actions
        self.parameterized_actions = {
            ActionType.INSTALL_PACKAGE: ['package_name'],
            ActionType.MODIFY_CODE: ['file', 'line', 'change'],
            ActionType.MODIFY_LR: ['value'],
            ActionType.MODIFY_BATCH: ['value'],
            ActionType.MODIFY_OPTIMIZER: ['value'],
            ActionType.MODIFY_EPOCHS: ['value'],
        }
    
    @property
    def n(self) -> int:
        """Number of actions."""
        return len(self.actions)
    
    def get_action_by_id(self, action_id: int) -> ActionType:
        """Get action type from ID."""
        return self.id_to_action[action_id]
    
    def get_id_by_action(self, action_type: ActionType) -> int:
        """Get ID from action type."""
        return self.action_to_id[action_type]
    
    def create_action(
        self,
        action_type: ActionType,
        parameters: Dict[str, Any] = None,
        reasoning: str = ""
    ) -> Action:
        """Create action with parameters."""
        return Action(
            action_type=action_type,
            parameters=parameters or {},
            reasoning=reasoning
        )
    
    def get_action_description(self, action_type: ActionType) -> str:
        """Get human-readable description."""
        descriptions = {
            ActionType.PARSE_PDF: "Parse PDF and extract text",
            ActionType.EXTRACT_GITHUB: "Extract GitHub links from paper",
            ActionType.CLONE_REPO: "Clone GitHub repository",
            ActionType.READ_README: "Read and parse README",
            ActionType.INSTALL_REQUIREMENTS: "Install requirements.txt",
            ActionType.RUN_TRAINING: "Execute training script",
            ActionType.ANALYZE_ERROR: "Analyze error message",
            ActionType.APPLY_FIX: "Apply code fix",
            ActionType.RUN_EXPERIMENT: "Run experiment with config",
            ActionType.COMPARE_RESULTS: "Compare results to paper",
        }
        return descriptions.get(action_type, action_type.value)
    
    def get_valid_actions(self, phase: str) -> List[ActionType]:
        """Get valid actions for current phase."""
        phase_actions = {
            'parsing': [
                ActionType.PARSE_PDF,
                ActionType.EXTRACT_GITHUB,
                ActionType.EXTRACT_METRICS,
                ActionType.VALIDATE_PARSING
            ],
            'repo_analysis': [
                ActionType.CLONE_REPO,
                ActionType.READ_README,
                ActionType.ANALYZE_CODE,
                ActionType.FIND_ENTRY_POINT,
                ActionType.EXTRACT_DEPS
            ],
            'setup': [
                ActionType.CREATE_VENV,
                ActionType.INSTALL_REQUIREMENTS,
                ActionType.DOWNLOAD_DATA,
                ActionType.VERIFY_SETUP
            ],
            'execution': [
                ActionType.RUN_TRAINING,
                ActionType.RUN_EVAL,
                ActionType.CHECK_LOGS
            ],
            'debugging': [
                ActionType.ANALYZE_ERROR,
                ActionType.SEARCH_SOLUTION,
                ActionType.APPLY_FIX,
                ActionType.TEST_FIX,
                ActionType.ROLLBACK
            ],
            'experimentation': [
                ActionType.MODIFY_LR,
                ActionType.MODIFY_BATCH,
                ActionType.MODIFY_OPTIMIZER,
                ActionType.RUN_EXPERIMENT
            ]
        }
        return phase_actions.get(phase, list(ActionType))


# Action templates for common patterns
class ActionTemplates:
    """Predefined action sequences for common scenarios."""
    
    @staticmethod
    def basic_setup_sequence() -> List[ActionType]:
        """Standard setup sequence."""
        return [
            ActionType.PARSE_PDF,
            ActionType.EXTRACT_GITHUB,
            ActionType.CLONE_REPO,
            ActionType.READ_README,
            ActionType.INSTALL_REQUIREMENTS,
            ActionType.VERIFY_SETUP
        ]
    
    @staticmethod
    def debugging_sequence() -> List[ActionType]:
        """Standard debugging sequence."""
        return [
            ActionType.ANALYZE_ERROR,
            ActionType.SEARCH_SOLUTION,
            ActionType.APPLY_FIX,
            ActionType.TEST_FIX
        ]
    
    @staticmethod
    def experimentation_sequence() -> List[ActionType]:
        """Standard experimentation sequence."""
        return [
            ActionType.MODIFY_LR,
            ActionType.RUN_EXPERIMENT,
            ActionType.COMPARE_RESULTS
        ]
