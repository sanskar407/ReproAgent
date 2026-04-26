import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)
    # Note: User specified gemini-2.5-flash, but we'll fallback to 1.5-flash if needed
    try:
        return genai.GenerativeModel('gemini-2.5-flash')
    except:
        return genai.GenerativeModel('gemini-pro')

def generate_summary_and_ppt_content(text: str):
    """
    Generates a summary and PPT structure from research paper text.
    """
    model = get_gemini_client()
    
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

    Format your response as a valid JSON object. Ensure all strings (especially the 'description') are properly escaped for JSON (e.g., use \\n for newlines).

    JSON structure:
    {{
        "description": "The summary following the formatting rules above",
        "ppt_slides": [
            {{
                "title": "Slide Title",
                "content": ["Key point 1", "Key point 2", ...]
            }}
        ]
    }}

    Research Paper Text:
    {text[:30000]}
    """


    response = model.generate_content(prompt)
    
    try:
        # Clean the response to ensure it's valid JSON
        content = response.text.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        
        # Use strict=False to be more lenient with control characters
        return json.loads(content, strict=False)
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return {
            "description": "Error generating description. Please try again.",
            "ppt_slides": []
        }

def analyze_installation_error(error_log: str, repo_structure: str):
    """
    Uses AI to analyze an installation error and suggest a fix.
    """
    model = get_gemini_client()
    
    prompt = f"""
    You are an expert DevOps and ML Engineer. A Python environment installation failed with the following error:
    
    ERROR LOG:
    {error_log[-2000:]}
    
    REPOSITORY STRUCTURE:
    {repo_structure}
    
    Based on the error, provide a solution to fix the installation. 
    Format your response as a JSON object:
    {{
        "diagnosis": "Short explanation of what went wrong",
        "action": "install_package" | "edit_requirements" | "change_python_version",
        "command": "The exact command to run to fix it (if any)",
        "file_to_edit": "path/to/file (if any)",
        "new_content": "New content for the file (if any)"
    }}
    """
    
    response = model.generate_content(prompt)
    try:
        content = response.text.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        return json.loads(content, strict=False)
    except:
        return None

def extract_execution_instructions(repo_structure: str, readme_text: str):
    """
    Asks AI to figure out how to run the evaluation/test script.
    """
    model = get_gemini_client()
    prompt = f"""
    Based on the repository structure and README, what is the exact command to run the evaluation or test script to verify the results?
    
    STRUCTURE:
    {repo_structure}
    
    README SNIPPET:
    {readme_text[:5000]}
    
    Return a JSON object:
    {{
        "command": "python eval.py ...",
        "explanation": "Why this command is selected"
    }}
    """
    response = model.generate_content(prompt)
    try:
        content = response.text.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        return json.loads(content, strict=False)
    except:
        return {{"command": "python main.py", "explanation": "Fallback to main.py"}}

def extract_claimed_metrics(paper_text: str):
    """
    Extracts the main results reported in the paper.
    """
    model = get_gemini_client()
    prompt = f"""
    Extract the primary performance metrics (accuracy, F1, FID, etc.) reported in the following paper text. 
    Focus on the main results table.
    
    TEXT:
    {paper_text[:20000]}
    
    Return a JSON object:
    {{
        "metrics": [
            {{"name": "Accuracy", "value": "94.2%", "context": "ImageNet validation"}},
            ...
        ]
    }}
    """
    response = model.generate_content(prompt)
    try:
        content = response.text.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        return json.loads(content, strict=False)
    except:
        return {{"metrics": []}}

def extract_metrics_from_logs(logs: str):
    """
    Parses execution logs to find resulting metrics.
    """
    model = get_gemini_client()
    prompt = f"""
    The following is the output log of a research paper's evaluation script. 
    Identify and extract the final performance metrics achieved.
    
    LOGS:
    {logs[-5000:]}
    
    Return a JSON object:
    {{
        "metrics": [
            {{"name": "Accuracy", "value": "93.8%"}},
            ...
        ]
    }}
    """
    response = model.generate_content(prompt)
    try:
        content = response.text.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        return json.loads(content, strict=False)
    except:
        return {{"metrics": []}}
