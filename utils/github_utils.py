"""
GitHub utilities for repository operations.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List


def clone_repository(
    repo_url: str,
    target_dir: Optional[str] = None,
    depth: int = 1,
    timeout: int = 60
) -> Optional[str]:
    """
    Clone a GitHub repository.
    
    Args:
        repo_url: GitHub repository URL
        target_dir: Target directory (optional, creates temp if None)
        depth: Clone depth (1 for shallow clone)
        timeout: Timeout in seconds
        
    Returns:
        Path to cloned repository, or None if failed
    """
    try:
        # Create target directory
        if target_dir is None:
            target_dir = tempfile.mkdtemp(prefix="reproagent_repo_")
        else:
            Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Cloning {repo_url} to {target_dir}...")
        
        # Clone with git
        cmd = ['git', 'clone', '--depth', str(depth), repo_url, target_dir]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"✅ Repository cloned successfully")
            return target_dir
        else:
            print(f"❌ Clone failed: {result.stderr}")
            return None
    
    except subprocess.TimeoutExpired:
        print(f"❌ Clone timeout after {timeout}s")
        return None
    
    except Exception as e:
        print(f"❌ Clone error: {e}")
        return None


def get_repo_info(repo_path: str) -> Dict[str, Any]:
    """
    Get information about a git repository.
    
    Args:
        repo_path: Path to repository
        
    Returns:
        Dictionary with repo info
    """
    info = {
        'path': repo_path,
        'exists': False,
        'is_git_repo': False,
        'remote_url': None,
        'branch': None,
        'last_commit': None,
        'file_count': 0,
        'size_mb': 0
    }
    
    repo_dir = Path(repo_path)
    
    if not repo_dir.exists():
        return info
    
    info['exists'] = True
    
    # Check if git repo
    git_dir = repo_dir / '.git'
    if not git_dir.exists():
        return info
    
    info['is_git_repo'] = True
    
    # Get remote URL
    try:
        result = subprocess.run(
            ['git', '-C', repo_path, 'config', '--get', 'remote.origin.url'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info['remote_url'] = result.stdout.strip()
    except:
        pass
    
    # Get current branch
    try:
        result = subprocess.run(
            ['git', '-C', repo_path, 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()
    except:
        pass
    
    # Get last commit
    try:
        result = subprocess.run(
            ['git', '-C', repo_path, 'log', '-1', '--pretty=format:%H %s'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            info['last_commit'] = result.stdout.strip()
    except:
        pass
    
    # Count files
    try:
        file_count = sum(1 for _ in repo_dir.rglob('*') if _.is_file())
        info['file_count'] = file_count
    except:
        pass
    
    # Calculate size
    try:
        total_size = sum(f.stat().st_size for f in repo_dir.rglob('*') if f.is_file())
        info['size_mb'] = total_size / (1024 * 1024)
    except:
        pass
    
    return info


def extract_github_urls(text: str) -> List[str]:
    """
    Extract GitHub URLs from text using regex.
    
    Args:
        text: Text to search
        
    Returns:
        List of GitHub URLs
    """
    pattern = r'https?://github\.com/[\w\-]+/[\w\-.]+'
    matches = re.findall(pattern, text)
    
    # Remove duplicates and clean
    urls = []
    for url in matches:
        # Remove trailing punctuation
        url = re.sub(r'[.,;)\]]+$', '', url)
        if url not in urls:
            urls.append(url)
    
    return urls


def parse_github_url(url: str) -> Optional[Dict[str, str]]:
    """
    Parse GitHub URL into components.
    
    Args:
        url: GitHub URL
        
    Returns:
        Dict with owner, repo, etc., or None if invalid
    """
    pattern = r'https?://github\.com/(?P<owner>[\w\-]+)/(?P<repo>[\w\-\.]+)'
    match = re.match(pattern, url)
    
    if match:
        return {
            'owner': match.group('owner'),
            'repo': match.group('repo'),
            'url': url
        }
    
    return None


def find_python_files(repo_path: str) -> List[str]:
    """
    Find all Python files in repository.
    
    Args:
        repo_path: Path to repository
        
    Returns:
        List of Python file paths (relative)
    """
    repo_dir = Path(repo_path)
    
    if not repo_dir.exists():
        return []
    
    python_files = []
    
    for py_file in repo_dir.rglob('*.py'):
        # Skip hidden directories and common non-code dirs
        parts = py_file.parts
        if any(p.startswith('.') or p in ['__pycache__', 'venv', 'env', 'build', 'dist'] for p in parts):
            continue
        
        rel_path = py_file.relative_to(repo_dir)
        python_files.append(str(rel_path))
    
    return python_files


def find_config_files(repo_path: str) -> Dict[str, Optional[str]]:
    """
    Find common configuration files.
    
    Args:
        repo_path: Path to repository
        
    Returns:
        Dict mapping config type to path
    """
    repo_dir = Path(repo_path)
    
    config_files = {
        'requirements': None,
        'setup': None,
        'pyproject': None,
        'dockerfile': None,
        'readme': None,
        'license': None
    }
    
    if not repo_dir.exists():
        return config_files
    
    # Check for each type
    if (repo_dir / 'requirements.txt').exists():
        config_files['requirements'] = 'requirements.txt'
    
    if (repo_dir / 'setup.py').exists():
        config_files['setup'] = 'setup.py'
    
    if (repo_dir / 'pyproject.toml').exists():
        config_files['pyproject'] = 'pyproject.toml'
    
    if (repo_dir / 'Dockerfile').exists():
        config_files['dockerfile'] = 'Dockerfile'
    
    # README (check multiple variants)
    for readme_name in ['README.md', 'README.rst', 'README.txt', 'README']:
        if (repo_dir / readme_name).exists():
            config_files['readme'] = readme_name
            break
    
    # LICENSE
    for license_name in ['LICENSE', 'LICENSE.md', 'LICENSE.txt']:
        if (repo_dir / license_name).exists():
            config_files['license'] = license_name
            break
    
    return config_files


# Test
if __name__ == "__main__":
    # Test URL extraction
    test_text = """
    Check out our code at https://github.com/example/awesome-repo
    Also see https://github.com/another/project.
    """
    
    urls = extract_github_urls(test_text)
    print("Found URLs:", urls)
    
    for url in urls:
        parsed = parse_github_url(url)
        print(f"Parsed: {parsed}")
