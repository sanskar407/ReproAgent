"""
Debugging agent - analyzes and fixes code errors.
"""

import re
from typing import Dict, Any, List, Optional, Tuple

from reproagent.models import LLMClient


class Debugger:
    """
    Debugging agent that:
    1. Analyzes error messages
    2. Searches for solutions
    3. Proposes fixes
    4. Applies patches
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Args:
            llm_client: LLM for error analysis
        """
        self.llm = llm_client
        
        # Common error patterns
        self.error_patterns = {
            'ImportError': r'ImportError: No module named [\'"](.+)[\'"]',
            'ModuleNotFoundError': r'ModuleNotFoundError: No module named [\'"](.+)[\'"]',
            'FileNotFoundError': r'FileNotFoundError: \[Errno 2\] No such file or directory: [\'"](.+)[\'"]',
            'RuntimeError': r'RuntimeError: (.+)',
            'ValueError': r'ValueError: (.+)',
            'TypeError': r'TypeError: (.+)',
            'AttributeError': r'AttributeError: (.+)',
        }
    
    def analyze_error(self, error_message: str, code_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze error and determine cause.
        
        Args:
            error_message: Full error message/traceback
            code_context: Relevant code snippet (optional)
            
        Returns:
            Analysis dict with error type, cause, and suggested fixes
        """
        print(f"🔍 Analyzing error...")
        
        # Classify error type
        error_type = self._classify_error(error_message)
        
        # Extract error details
        error_details = self._extract_error_details(error_message, error_type)
        
        # Get LLM analysis
        llm_analysis = self._llm_analyze_error(error_message, code_context)
        
        analysis = {
            'error_type': error_type,
            'error_details': error_details,
            'root_cause': llm_analysis.get('root_cause', 'Unknown'),
            'suggested_fixes': llm_analysis.get('fixes', []),
            'confidence': llm_analysis.get('confidence', 0.5)
        }
        
        print(f"✅ Error analyzed: {error_type}")
        print(f"   Cause: {analysis['root_cause']}")
        
        return analysis
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type."""
        for error_type, pattern in self.error_patterns.items():
            if re.search(pattern, error_message):
                return error_type
        
        # Check for common error types in message
        if 'import' in error_message.lower():
            return 'ImportError'
        elif 'file' in error_message.lower() and 'not found' in error_message.lower():
            return 'FileNotFoundError'
        elif 'cuda' in error_message.lower() or 'gpu' in error_message.lower():
            return 'CUDAError'
        elif 'memory' in error_message.lower():
            return 'MemoryError'
        
        return 'UnknownError'
    
    def _extract_error_details(self, error_message: str, error_type: str) -> Dict[str, str]:
        """Extract specific details from error."""
        details = {}
        
        if error_type in self.error_patterns:
            pattern = self.error_patterns[error_type]
            match = re.search(pattern, error_message)
            if match:
                details['detail'] = match.group(1)
        
        # Extract file and line number
        file_pattern = r'File "(.+)", line (\d+)'
        file_match = re.search(file_pattern, error_message)
        if file_match:
            details['file'] = file_match.group(1)
            details['line'] = file_match.group(2)
        
        return details
    
    def _llm_analyze_error(self, error_message: str, code_context: Optional[str]) -> Dict[str, Any]:
        """Use LLM to analyze error."""
        
        prompt = f"""
Analyze this Python error and provide solutions.

Error:
{error_message[:1000]}
"""
        
        if code_context:
            prompt += f"\n\nRelevant code:\n{code_context[:500]}"
        
        prompt += """

Respond with JSON:
{
    "root_cause": "explanation of what caused the error",
    "fixes": ["fix 1", "fix 2", "fix 3"],
    "confidence": 0.9
}
"""
        
        try:
            result = self.llm.generate_structured(prompt)
            return result
        except:
            return self._fallback_analysis(error_message)
    
    def _fallback_analysis(self, error_message: str) -> Dict[str, Any]:
        """Fallback analysis without LLM."""
        
        # Common fixes for common errors
        fixes = []
        
        if 'ModuleNotFoundError' in error_message or 'ImportError' in error_message:
            match = re.search(r"module named ['\"](.+)['\"]", error_message)
            if match:
                module = match.group(1)
                fixes = [
                    f"Install missing package: pip install {module}",
                    f"Check if {module} is in requirements.txt",
                    "Activate correct virtual environment"
                ]
        
        elif 'FileNotFoundError' in error_message:
            fixes = [
                "Check if file path is correct",
                "Ensure data is downloaded",
                "Check working directory"
            ]
        
        elif 'CUDA' in error_message or 'GPU' in error_message:
            fixes = [
                "Check CUDA installation",
                "Verify GPU availability",
                "Try running on CPU: device='cpu'"
            ]
        
        elif 'memory' in error_message.lower():
            fixes = [
                "Reduce batch size",
                "Use gradient accumulation",
                "Clear GPU cache: torch.cuda.empty_cache()"
            ]
        
        return {
            'root_cause': 'Error detected',
            'fixes': fixes or ['Debug manually'],
            'confidence': 0.6
        }
    
    def generate_fix(self, error_analysis: Dict[str, Any]) -> str:
        """
        Generate code fix based on error analysis.
        
        Args:
            error_analysis: Output from analyze_error()
            
        Returns:
            Fix as code or command
        """
        error_type = error_analysis['error_type']
        details = error_analysis['error_details']
        
        # Generate specific fix based on error type
        if error_type in ['ImportError', 'ModuleNotFoundError']:
            module = details.get('detail', '')
            return f"pip install {module}"
        
        elif error_type == 'FileNotFoundError':
            file_path = details.get('detail', '')
            return f"# Check if {file_path} exists or download it"
        
        elif error_type == 'CUDAError':
            return "# Try: model.to('cpu') or install CUDA"
        
        elif error_type == 'MemoryError':
            return "# Reduce batch_size or use gradient accumulation"
        
        # Use LLM for complex fixes
        return self._llm_generate_fix(error_analysis)
    
    def _llm_generate_fix(self, error_analysis: Dict[str, Any]) -> str:
        """Use LLM to generate code fix."""
        
        prompt = f"""
Generate a code fix for this error:

Error Type: {error_analysis['error_type']}
Root Cause: {error_analysis['root_cause']}

Provide the fix as Python code or shell command.
"""
        
        try:
            fix = self.llm.generate(prompt, max_tokens=200)
            return fix.strip()
        except:
            return "# Manual fix required"
    
    def search_solution(self, error_message: str) -> List[str]:
        """
        Search for solutions to error.
        Simulates searching StackOverflow, documentation, etc.
        
        Args:
            error_message: Error message
            
        Returns:
            List of solution suggestions
        """
        # In full implementation, would search:
        # - StackOverflow API
        # - GitHub Issues
        # - Documentation
        
        # For now, use LLM to generate solutions
        prompt = f"""
This error occurred: {error_message[:500]}

List 3 common solutions to this error.
Respond with JSON:
{{
    "solutions": ["solution 1", "solution 2", "solution 3"]
}}
"""
        
        try:
            result = self.llm.generate_structured(prompt)
            return result.get('solutions', [])
        except:
            return ["Check dependencies", "Review code", "Search documentation"]


# Test
if __name__ == "__main__":
    from reproagent.models import LLMClient
    
    llm = LLMClient()
    debugger = Debugger(llm)
    
    # Test error
    error = """
Traceback (most recent call last):
  File "train.py", line 10, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
"""
    
    analysis = debugger.analyze_error(error)
    print(analysis)
    
    fix = debugger.generate_fix(analysis)
    print(f"\nFix: {fix}")
