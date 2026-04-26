import os
import json
from dotenv import load_dotenv

load_dotenv()

def _call_gemini(prompt: str) -> str:
    """
    Calls the Gemini API using the new google.genai SDK.
    Falls back gracefully if the API key is missing.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please add it in HF Space Settings > Secrets.")

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
    except ImportError:
        # Fallback to old SDK if new one not available
        import google.generativeai as genai_old
        genai_old.configure(api_key=api_key)
        model = genai_old.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text


def _parse_json(text: str) -> dict:
    """Strips markdown fences and parses JSON."""
    content = text.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return json.loads(content.strip(), strict=False)


def generate_summary_and_ppt_content(text: str) -> dict:
    """
    Generates a summary and PPT structure from research paper text using Gemini.
    """
    prompt = f"""
    Analyze the research paper and provide two things:
    1. A summary in a clean, structured format.
    2. A structured plan for an impressive PowerPoint presentation.

    STRICT FORMATTING RULES FOR THE SUMMARY:
    - Use clear section headings like: 1. Core Idea, 2. Background, etc.
    - Do NOT use emojis.
    - Do NOT use excessive bold formatting inside paragraphs.
    - Only bold the section titles.
    - Use bullet points (•) instead of long paragraphs.
    - Keep sentences short and clear.
    - Avoid decorative or marketing-style language.
    - Keep it concise but informative.
    - Do not use * at all.

    SUMMARY STRUCTURE:
    1. Core Idea
    2. Background / Problem
    3. Key Observation
    4. Method (How it works)
    5. Results
    6. Contributions
    7. Limitations (if any)

    Format your response as a valid JSON object with this structure:
    {{
        "description": "The full summary following the formatting rules above",
        "slides": [
            {{
                "title": "Slide Title",
                "content": ["Key point 1", "Key point 2", "Key point 3"]
            }}
        ]
    }}

    Research Paper Text:
    {text[:30000]}
    """

    try:
        raw = _call_gemini(prompt)
        return _parse_json(raw)
    except Exception as e:
        print(f"[LLM ERROR] generate_summary_and_ppt_content: {e}")
        return {
            "description": "Could not generate summary. Please check your GEMINI_API_KEY.",
            "slides": [
                {"title": "Error", "content": [str(e)]}
            ]
        }


def analyze_installation_error(error_log: str, repo_structure: str) -> dict:
    """Uses Gemini to analyze an installation error and suggest a fix."""
    prompt = f"""
    You are an expert DevOps and ML Engineer. A Python environment installation failed.

    ERROR LOG:
    {error_log[-2000:]}

    REPOSITORY STRUCTURE:
    {repo_structure}

    Return a JSON object:
    {{
        "diagnosis": "Short explanation of what went wrong",
        "action": "install_package",
        "command": "pip install ...",
        "file_to_edit": "",
        "new_content": ""
    }}
    """
    try:
        raw = _call_gemini(prompt)
        return _parse_json(raw)
    except Exception as e:
        print(f"[LLM ERROR] analyze_installation_error: {e}")
        return {"diagnosis": str(e), "action": "manual", "command": "", "file_to_edit": "", "new_content": ""}


def extract_execution_instructions(repo_structure: str, readme_text: str) -> dict:
    """Asks Gemini to figure out how to run the evaluation script."""
    prompt = f"""
    Based on the repository structure and README, what is the exact command to run the evaluation?

    STRUCTURE:
    {repo_structure}

    README:
    {readme_text[:5000]}

    Return JSON:
    {{
        "command": "python eval.py ...",
        "explanation": "Why this command"
    }}
    """
    try:
        raw = _call_gemini(prompt)
        return _parse_json(raw)
    except Exception as e:
        print(f"[LLM ERROR] extract_execution_instructions: {e}")
        return {"command": "python main.py", "explanation": "Fallback"}


def extract_claimed_metrics(paper_text: str) -> dict:
    """Extracts the main results reported in the paper."""
    prompt = f"""
    Extract the primary performance metrics (accuracy, F1, FID, etc.) from the paper text.

    TEXT:
    {paper_text[:20000]}

    Return JSON:
    {{
        "metrics": [
            {{"name": "Accuracy", "value": "94.2%", "context": "ImageNet validation"}}
        ]
    }}
    """
    try:
        raw = _call_gemini(prompt)
        return _parse_json(raw)
    except Exception as e:
        print(f"[LLM ERROR] extract_claimed_metrics: {e}")
        return {"metrics": []}


def extract_metrics_from_logs(logs: str) -> dict:
    """Parses execution logs to find resulting metrics."""
    prompt = f"""
    From the following evaluation log, extract the final performance metrics.

    LOGS:
    {logs[-5000:]}

    Return JSON:
    {{
        "metrics": [
            {{"name": "Accuracy", "value": "93.8%"}}
        ]
    }}
    """
    try:
        raw = _call_gemini(prompt)
        return _parse_json(raw)
    except Exception as e:
        print(f"[LLM ERROR] extract_metrics_from_logs: {e}")
        return {"metrics": []}
