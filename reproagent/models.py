"""
LLM model client wrapper.
Supports multiple providers with unified interface.
"""

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    Auto-detects available API keys.
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Args:
            provider: 'gemini', 'groq', 'openai', 'huggingface', or None (auto-detect)
        """
        self.provider = provider or self._detect_provider()
        self.client = self._init_client()
        
        try:
            print(f"\U0001f916 LLM initialized: {self.provider}")
        except UnicodeEncodeError:
            print(f"[LLM] Initialized: {self.provider}")
    
    def _detect_provider(self) -> str:
        """Auto-detect available API."""
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        elif os.getenv("GROQ_API_KEY"):
            return "groq"
        elif os.getenv("OPENAI_API_KEY"):
            return "openai"
        elif os.getenv("HF_TOKEN"):
            return "huggingface"
        else:
            print("[WARN] No API key found, using mock mode")
            return "mock"
    
    def _init_client(self):
        """Initialize provider client."""
        if self.provider == "gemini":
            try:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                return genai.GenerativeModel('gemini-2.5-flash')
            except ImportError:
                print("[WARN] google-generativeai not installed, falling back to mock")
                return None
        elif self.provider == "groq":
            try:
                from groq import Groq
                return Groq(api_key=os.getenv("GROQ_API_KEY"))
            except ImportError:
                print("[WARN] Groq not installed, falling back to mock")
                return None
        
        elif self.provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                print("[WARN] OpenAI not installed, falling back to mock")
                return None
        
        elif self.provider == "huggingface":
            try:
                from huggingface_hub import InferenceClient
                return InferenceClient(token=os.getenv("HF_TOKEN"))
            except ImportError:
                print("[WARN] HuggingFace not installed, falling back to mock")
                return None
        
        return None
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """Generate text completion."""
        
        if self.provider == "mock" or self.client is None:
            return self._mock_generate(prompt)
        
        try:
            if self.provider == "gemini":
                response = self.client.generate_content(
                    f"{system_prompt}\n\n{prompt}",
                    generation_config={"max_output_tokens": max_tokens, "temperature": temperature}
                )
                return response.text
            
            elif self.provider == "groq":
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            
            elif self.provider == "huggingface":
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                response = self.client.text_generation(
                    "Qwen/Qwen2.5-72B-Instruct",
                    prompt=full_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                return response
        
        except Exception as e:
            print(f"[ERROR] LLM Error: {e}")
            return self._mock_generate(prompt)
    
    def _mock_generate(self, prompt: str) -> str:
        """Mock generation for testing without API."""
        if "github" in prompt.lower():
            return "https://github.com/example/repo"
        elif "error" in prompt.lower():
            return "The error is likely due to missing dependencies. Try installing the required packages."
        elif "metric" in prompt.lower():
            return "Target accuracy: 95%"
        else:
            return f"Mock response for: {prompt[:50]}..."
    
    def generate_structured(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> Dict[str, Any]:
        """Generate structured JSON response."""
        import json
        
        json_prompt = f"{prompt}\n\nRespond ONLY with valid JSON."
        response = self.generate(json_prompt, system_prompt)
        
        try:
            # Extract JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                return {"error": "No JSON found", "raw": response}
        except Exception as e:
            return {"error": str(e), "raw": response}
