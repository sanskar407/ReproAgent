"""
Repository analyzer - analyzes GitHub repositories.
"""

import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import subprocess

from reproagent.models import LLMClient
from reproagent.state import RepoState


class RepoAnalyzer:
    """
    Analyzes GitHub repositories to understand:
    - Code structure
    - Dependencies
    - Entry points
    - Setup instructions
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Args:
            llm_client: LLM for code analysis
        """
        self.llm = llm_client
    
    def analyze_repo(self, repo_url: str, local_path: Optional[str] = None) -> RepoState:
        """
        Analyze a GitHub repository.
        
        Args:
            repo_url: GitHub URL
            local_path: Local path (if already cloned)
            
        Returns:
            RepoState with analysis
        """
        print(f"📦 Analyzing repository: {repo_url}")
        
        # Clone if needed
        if not local_path:
            local_path = self._clone_repo(repo_url)
        
        if not local_path or not Path(local_path).exists():
            print(f"❌ Failed to access repository")
            return RepoState(url=repo_url)
        
        # Analyze components
        readme_content = self._read_readme(local_path)
        dependencies = self._extract_dependencies(local_path)
        entry_point = self._find_entry_point(local_path)
        framework = self._detect_framework(local_path, dependencies)
        setup_instructions = self._extract_setup_instructions(readme_content)
        
        state = RepoState(
            url=repo_url,
            cloned=True,
            local_path=local_path,
            readme_content=readme_content,
            setup_instructions=setup_instructions,
            dependencies=dependencies,
            entry_point=entry_point,
            framework=framework,
            repo_quality_score=self._calculate_quality_score(local_path, readme_content)
        )
        
        print(f"✅ Repository analyzed")
        print(f"   Framework: {state.framework}")
        print(f"   Entry point: {state.entry_point}")
        print(f"   Dependencies: {len(state.dependencies)}")
        
        return state
    
    def _clone_repo(self, repo_url: str) -> Optional[str]:
        """
        Clone GitHub repository.
        
        Args:
            repo_url: GitHub URL
            
        Returns:
            Local path or None if failed
        """
        try:
            # Create temp directory
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="reproagent_")
            
            print(f"📥 Cloning to {temp_dir}...")
            
            # Clone with git
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', repo_url, temp_dir],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✅ Repository cloned")
                return temp_dir
            else:
                print(f"❌ Clone failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Clone error: {e}")
            return None
    
    def _read_readme(self, repo_path: str) -> str:
        """Read README file."""
        readme_files = ['README.md', 'README.rst', 'README.txt', 'README']
        
        for readme_name in readme_files:
            readme_path = Path(repo_path) / readme_name
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    print(f"⚠️  Error reading {readme_name}: {e}")
        
        return ""
    
    def _extract_dependencies(self, repo_path: str) -> List[str]:
        """Extract dependencies from requirements files."""
        dependencies = []
        
        # Check requirements.txt
        req_path = Path(repo_path) / 'requirements.txt'
        if req_path.exists():
            try:
                with open(req_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Extract package name (before ==, >=, etc.)
                            pkg = re.split(r'[=<>!]', line)[0].strip()
                            dependencies.append(pkg)
            except Exception as e:
                print(f"⚠️  Error reading requirements.txt: {e}")
        
        # Check setup.py
        setup_path = Path(repo_path) / 'setup.py'
        if setup_path.exists():
            try:
                with open(setup_path, 'r') as f:
                    content = f.read()
                    # Look for install_requires
                    match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                    if match:
                        deps_str = match.group(1)
                        for dep in re.findall(r'["\']([^"\']+)["\']', deps_str):
                            pkg = re.split(r'[=<>!]', dep)[0].strip()
                            if pkg not in dependencies:
                                dependencies.append(pkg)
            except Exception as e:
                print(f"⚠️  Error reading setup.py: {e}")
        
        # Check pyproject.toml
        pyproject_path = Path(repo_path) / 'pyproject.toml'
        if pyproject_path.exists():
            try:
                import tomli
                with open(pyproject_path, 'rb') as f:
                    data = tomli.load(f)
                    deps = data.get('project', {}).get('dependencies', [])
                    for dep in deps:
                        pkg = re.split(r'[=<>!]', dep)[0].strip()
                        if pkg not in dependencies:
                            dependencies.append(pkg)
            except:
                pass
        
        return dependencies
    
    def _find_entry_point(self, repo_path: str) -> str:
        """Find main entry point script."""
        # Common entry point names
        candidates = [
            'train.py',
            'main.py',
            'run.py',
            'train_model.py',
            'finetune.py',
            'run_training.py'
        ]
        
        repo_dir = Path(repo_path)
        
        for candidate in candidates:
            if (repo_dir / candidate).exists():
                return candidate
        
        # Search in subdirectories
        for py_file in repo_dir.rglob('*.py'):
            if py_file.stem in ['train', 'main', 'run']:
                return str(py_file.relative_to(repo_dir))
        
        return ""
    
    def _detect_framework(self, repo_path: str, dependencies: List[str]) -> str:
        """Detect ML framework used."""
        dep_str = ' '.join(dependencies).lower()
        
        if 'torch' in dep_str or 'pytorch' in dep_str:
            return 'pytorch'
        elif 'tensorflow' in dep_str or 'tf' in dep_str:
            return 'tensorflow'
        elif 'jax' in dep_str:
            return 'jax'
        elif 'keras' in dep_str:
            return 'keras'
        
        # Check imports in Python files
        try:
            for py_file in Path(repo_path).rglob('*.py'):
                with open(py_file, 'r') as f:
                    content = f.read(1000)  # First 1000 chars
                    if 'import torch' in content:
                        return 'pytorch'
                    elif 'import tensorflow' in content:
                        return 'tensorflow'
        except:
            pass
        
        return "unknown"
    
    def _extract_setup_instructions(self, readme_content: str) -> List[str]:
        """
        Extract setup instructions from README using LLM.
        
        Args:
            readme_content: README text
            
        Returns:
            List of setup steps
        """
        if not readme_content:
            return []
        
        # Truncate README
        readme_sample = readme_content[:3000]
        
        prompt = f"""
Extract step-by-step setup/installation instructions from this README.

README:
{readme_sample}

Respond with JSON:
{{
    "setup_steps": ["step 1", "step 2", ...]
}}
"""
        
        try:
            result = self.llm.generate_structured(prompt)
            return result.get('setup_steps', [])
        except:
            # Fallback: simple extraction
            return self._simple_setup_extraction(readme_content)
    
    def _simple_setup_extraction(self, readme: str) -> List[str]:
        """Simple regex-based setup extraction."""
        steps = []
        
        # Look for pip install commands
        pip_pattern = r'pip install (.+)'
        for match in re.finditer(pip_pattern, readme):
            steps.append(f"pip install {match.group(1).strip()}")
        
        # Look for numbered steps
        step_pattern = r'^\d+\.\s+(.+)$'
        for line in readme.split('\n'):
            match = re.match(step_pattern, line.strip())
            if match:
                steps.append(match.group(1))
        
        return steps[:10]  # Max 10 steps
    
    def _calculate_quality_score(self, repo_path: str, readme: str) -> float:
        """
        Calculate repository quality score.
        
        Factors:
        - Has README
        - Has requirements/setup files
        - Has tests
        - Code organization
        """
        score = 0.0
        
        # Has README (0.3)
        if readme:
            score += 0.3
        
        # Has requirements (0.2)
        if (Path(repo_path) / 'requirements.txt').exists():
            score += 0.2
        
        # Has setup.py or pyproject.toml (0.2)
        if (Path(repo_path) / 'setup.py').exists() or (Path(repo_path) / 'pyproject.toml').exists():
            score += 0.2
        
        # Has tests (0.15)
        if (Path(repo_path) / 'tests').exists() or (Path(repo_path) / 'test').exists():
            score += 0.15
        
        # Has LICENSE (0.05)
        if (Path(repo_path) / 'LICENSE').exists():
            score += 0.05
        
        # Has .gitignore (0.05)
        if (Path(repo_path) / '.gitignore').exists():
            score += 0.05
        
        # Good README length (0.05)
        if len(readme) > 500:
            score += 0.05
        
        return min(1.0, score)


# Test
if __name__ == "__main__":
    from reproagent.models import LLMClient
    
    llm = LLMClient()
    analyzer = RepoAnalyzer(llm)
    
    # Test with a real repo
    state = analyzer.analyze_repo("https://github.com/pytorch/examples")
    print(state.to_dict())
