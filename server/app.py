"""
ReproAgent - Gradio Web Interface
Interactive demo for AI-powered ML paper reproduction.

Three tabs:
  1. Reproduce a Paper — Upload PDF or paste URL, agent works through it live
  2. Simulation Demo  — Quick simulation with pre-loaded papers
  3. Benchmark        — Compare reasoning vs random agents
"""

import sys
import os
import re
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Generator

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import numpy as np

from reproagent.environment import ReproAgentEnv
from reproagent.state import PaperState
from reproagent.models import LLMClient
from reproagent.papers import create_sample_papers
from agents.reasoning_agent import create_agent


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def safe_print(msg: str):
    """Print without unicode crashes on Windows."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode())


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using available libraries."""
    # Try pdfplumber first
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:15]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except Exception:
        pass

    # Fallback to PyPDF2
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages[:15]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except Exception:
        pass

    return ""


def extract_paper_info_regex(text: str) -> Dict[str, Any]:
    """Regex-based extraction of paper metadata from raw text."""
    info: Dict[str, Any] = {
        "title": "",
        "abstract": "",
        "github_links": [],
        "datasets": [],
        "metrics": [],
        "key_claims": [],
    }

    # Title: first non-empty line that looks like a title
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        info["title"] = lines[0][:200]

    # Abstract
    abs_match = re.search(
        r"(?i)abstract[:\s]*\n?(.*?)(?:\n\s*\n|introduction|1[\.\s])",
        text, re.DOTALL,
    )
    if abs_match:
        info["abstract"] = abs_match.group(1).strip()[:1000]

    # GitHub links
    gh_urls = re.findall(r"https?://github\.com/[\w\-]+/[\w\-\.]+", text)
    # Clean trailing punctuation (period, comma, etc.) from each URL
    cleaned = []
    for url in gh_urls:
        url = re.sub(r'[.,;:)\]!?\'"]+$', '', url)  # strip trailing punctuation
        url = url.rstrip('.')  # extra safety for trailing dots
        if url not in cleaned:
            cleaned.append(url)
    info["github_links"] = cleaned

    # Datasets
    known_datasets = [
        "CIFAR-10", "CIFAR-100", "MNIST", "ImageNet", "COCO",
        "SST-2", "GLUE", "SQuAD", "WMT", "CelebA", "VOC",
    ]
    for ds in known_datasets:
        if ds.lower() in text.lower():
            info["datasets"].append(ds)

    # Metrics — look for common ML metrics with numbers
    metric_patterns = [
        r"(?i)(accuracy|acc)[\s:=]*(\d+\.?\d*)\s*%",
        r"(?i)(accuracy|acc)[\s:=]*(0\.\d+)",
        r"(?i)(f1[\s\-]?score)[\s:=]*(\d+\.?\d*)",
        r"(?i)(bleu)[\s:=]*(\d+\.?\d*)",
        r"(?i)(FID)[\s:=of ]*(\d+\.?\d*)",
        r"(?i)(perplexity|ppl)[\s:=]*(\d+\.?\d*)",
        r"(?i)(speedup|speed-up)[\s:of=]*(\d+\.?\d*)[x\s]",
        r"(?i)(MACs?|FLOPs?)[\s:=reduction of]*(\d+\.?\d*)%",
        r"(?i)(PSNR)[\s:=]*(\d+\.?\d*)",
        r"(?i)(SSIM)[\s:=]*(0\.\d+)",
        r"(?i)(mAP|AP)[\s:=]*(\d+\.?\d*)",
        r"(?i)(top-?1)[\s:=accuracy ]*(\d+\.?\d*)",
    ]
    for pat in metric_patterns:
        for m in re.finditer(pat, text):
            info["metrics"].append({"name": m.group(1), "value": m.group(2)})

    return info


def extract_paper_info_llm(text: str, llm: LLMClient) -> Dict[str, Any]:
    """Use Groq LLM to intelligently extract paper metadata."""
    prompt = f"""You are an expert ML research assistant. Extract the following from this research paper text:

1. title - Full paper title
2. abstract - The abstract (first 500 chars)
3. github_links - Any GitHub repository URLs mentioned
4. datasets - Datasets used (e.g., CIFAR-10, ImageNet)
5. target_metric_name - Main evaluation metric name (e.g. FID, CLIP score, BLEU, accuracy). Extract this EXACTLY as written in the text. DO NOT default to accuracy.
6. target_metric_value - The numerical claim for this metric (e.g. 7.5, 0.95). Extract EXACTLY as written. DO NOT normalize or guess.
7. model_name - The primary model architecture
8. key_claims - List of 3-5 key claims from the paper

Respond ONLY with valid JSON.

Paper text (first 3000 chars):
{text[:3000]}
"""
    try:
        result = llm.generate_structured(prompt)
        safe_print(f"[DEBUG] LLM raw result: {json.dumps(result)[:500]}")
        if "error" not in result:
            # Clean github links from LLM too
            gh_links = result.get("github_links", [])
            if isinstance(gh_links, str):
                gh_links = [gh_links] if gh_links else []
            gh_links = [re.sub(r'[.,;:)\]]+$', '', u).rstrip('.') for u in gh_links]
            
            return {
                "title": result.get("title", ""),
                "abstract": result.get("abstract", ""),
                "github_links": gh_links,
                "datasets": result.get("datasets", []) if isinstance(result.get("datasets"), list) else [result.get("datasets", "")],
                "metrics": [
                    {
                        "name": result.get("target_metric_name", "accuracy"),
                        "value": str(result.get("target_metric_value", "")),
                    }
                ] if result.get("target_metric_value") else [],
                "model_name": result.get("model_name", ""),
                "key_claims": result.get("key_claims", []) if isinstance(result.get("key_claims"), list) else [],
            }
        else:
            safe_print(f"[WARN] LLM returned error: {result.get('error')}")
    except Exception as e:
        safe_print(f"[WARN] LLM extraction failed: {e}")
        import traceback
        traceback.print_exc()

    return {}


# ---------------------------------------------------------------------------
#  Tab 1: Reproduce a Paper
# ---------------------------------------------------------------------------

def run_paper_reproduction(
    pdf_file,
    paper_url: str,
    use_llm: bool,
    max_steps: int,
    execution_mode: str,
    clone_dir: str,
) -> Generator:
    """
    Main reproduction pipeline.
    Yields (log_md, paper_info_md, metrics_md, state_json) as it progresses.
    """
    log_lines: List[str] = []

    def log(msg: str):
        log_lines.append(msg)
        return "\n".join(log_lines)

    empty = ("", "", "{}", "{}")

    # --- Step 0: Input validation ---
    if pdf_file is None and not paper_url.strip():
        yield (log("**Please upload a PDF or paste a paper URL.**"), "", "{}", "{}")
        return

    yield (log("### Starting ReproAgent...\n"), "", "{}", "{}")
    time.sleep(0.3)

    # --- Step 1: Get paper text ---
    paper_text = ""
    paper_title = ""

    if pdf_file is not None:
        pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
        yield (log(f"**Step 1/9: Reading PDF** `{Path(pdf_path).name}`..."), "", "{}", "{}")
        time.sleep(0.2)
        paper_text = extract_text_from_pdf(pdf_path)
        if not paper_text:
            yield (log("- Could not extract text from PDF. Is it a scanned image?"), "", "{}", "{}")
            return
        yield (log(f"- Extracted **{len(paper_text):,} characters** from PDF\n"), "", "{}", "{}")
    elif paper_url.strip():
        yield (log(f"**Step 1/9: Fetching paper** from `{paper_url.strip()[:80]}`..."), "", "{}", "{}")
        time.sleep(0.3)
        # Try to fetch URL content
        try:
            import requests
            resp = requests.get(paper_url.strip(), timeout=15)
            if resp.status_code == 200:
                if paper_url.strip().endswith(".pdf"):
                    # Save temp PDF and extract
                    tmp_path = Path("data/tmp_paper.pdf")
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path.write_bytes(resp.content)
                    paper_text = extract_text_from_pdf(str(tmp_path))
                else:
                    paper_text = resp.text[:10000]
                yield (log(f"- Fetched **{len(paper_text):,} characters**\n"), "", "{}", "{}")
            else:
                yield (log(f"- Failed to fetch URL (status {resp.status_code})\n"), "", "{}", "{}")
                return
        except Exception as e:
            yield (log(f"- Error fetching URL: {e}\n"), "", "{}", "{}")
            return

    # --- Step 2: Extract paper info ---
    yield (log("**Step 2/9: Analyzing paper content**..."), "", "{}", "{}")
    time.sleep(0.2)

    # Try LLM first, fallback to regex
    llm_client = None
    paper_info = {}
    if use_llm:
        try:
            llm_client = LLMClient()
            if llm_client.provider != "mock":
                yield (log(f"- Using **{llm_client.provider.upper()}** LLM for intelligent extraction"), "", "{}", "{}")
                paper_info = extract_paper_info_llm(paper_text, llm_client)
        except Exception:
            pass

    if not paper_info or not paper_info.get("title"):
        yield (log("- Using **regex** extraction (LLM unavailable or failed)"), "", "{}", "{}")
        paper_info = extract_paper_info_regex(paper_text)

    paper_title = paper_info.get("title", "Unknown Paper")
    github_links = paper_info.get("github_links", [])
    datasets = paper_info.get("datasets", [])
    metrics = paper_info.get("metrics", [])
    model_name = paper_info.get("model_name", "Unknown")
    key_claims = paper_info.get("key_claims", [])

    # Determine target metric
    target_metric = 0.0
    metric_name = "Unknown"
    if metrics:
        metric_name = metrics[0].get("name", "accuracy")
        try:
            val = float(metrics[0].get("value", "0.0"))
            target_metric = val
        except (ValueError, TypeError):
            target_metric = 0.95
    else:
        # Fallback if neither regex nor LLM generated any metrics
        target_metric = 0.95
        metric_name = "accuracy"

    # Build paper info markdown
    paper_info_md = f"""## Paper Information

| Field | Value |
|-------|-------|
| **Title** | {paper_title[:100]} |
| **Model** | {model_name} |
| **Dataset(s)** | {', '.join(datasets) if datasets else 'Not detected'} |
| **Target Metric** | {target_metric:.3f} ({metric_name}) |
| **GitHub Links** | {', '.join(f'[link]({u})' for u in github_links) if github_links else 'None found'} |

"""
    if key_claims:
        paper_info_md += "### Key Claims\n"
        for claim in key_claims[:5]:
            paper_info_md += f"- {claim}\n"

    yield (log(f"- Title: **{paper_title[:80]}**"), paper_info_md, "{}", "{}")
    time.sleep(0.2)
    yield (log(f"- Found **{len(github_links)}** GitHub link(s)"), paper_info_md, "{}", "{}")
    yield (log(f"- Target: **{target_metric:.3f}** ({metric_name})\n"), paper_info_md, "{}", "{}")

    # --- Step 3-9: Run agent loop via environment ---
    yield (log("**Step 3/9: Initializing reproduction environment**...\n"), paper_info_md, "{}", "{}")
    time.sleep(0.2)

    try:
        env = ReproAgentEnv(
            difficulty="easy",
            max_steps=int(max_steps),
            use_llm=use_llm,
            render_mode=None,
            exec_mode=execution_mode,
            workspace_dir=clone_dir.strip() if clone_dir.strip() else "/tmp/reproagent",
        )
        # Override paper state with what we extracted
        obs, info = env.reset()
        env.state.paper = PaperState(
            title=paper_title,
            dataset=datasets[0] if datasets else "Unknown",
            model=model_name,
            target_metric=target_metric,
            metric_name=metric_name,
            github_links=github_links,
            key_claims=key_claims,
            parsed=True,
            confidence=0.85,
        )
        env.state.experiment.target_metric = target_metric
        env.state.experiment.gap = target_metric

        agent = create_agent(env, agent_type="reasoning", use_llm=use_llm)
        agent.reset()

    except Exception as e:
        yield (log(f"\n**Error initializing:** {e}"), paper_info_md, "{}", "{}")
        return

    yield (log("- Environment ready. Starting agent loop...\n"), paper_info_md, "{}", "{}")

    step_labels = {
        "parse_pdf": ("Step 3/9", "Reading paper"),
        "extract_github": ("Step 4/9", "Finding GitHub repo"),
        "extract_metrics": ("Step 4/9", "Extracting metrics"),
        "validate_parsing": ("Step 4/9", "Validating parse"),
        "clone_repo": ("Step 5/9", "Cloning repository"),
        "read_readme": ("Step 5/9", "Reading README"),
        "analyze_code": ("Step 5/9", "Analyzing code structure"),
        "find_entry_point": ("Step 5/9", "Finding entry point"),
        "extract_deps": ("Step 5/9", "Extracting dependencies"),
        "create_venv": ("Step 6/9", "Creating environment"),
        "install_requirements": ("Step 6/9", "Installing dependencies"),
        "install_package": ("Step 6/9", "Installing package"),
        "download_data": ("Step 6/9", "Downloading data"),
        "verify_setup": ("Step 6/9", "Verifying setup"),
        "run_training": ("Step 7/9", "Running code"),
        "run_eval": ("Step 7/9", "Running evaluation"),
        "analyze_error": ("Step 7/9", "Debugging error"),
        "apply_fix": ("Step 7/9", "Applying fix"),
        "search_solution": ("Step 7/9", "Searching for solution"),
        "modify_code": ("Step 7/9", "Modifying code"),
        "test_fix": ("Step 7/9", "Testing fix"),
        "run_experiment": ("Step 8/9", "Tuning hyperparameters"),
        "modify_learning_rate": ("Step 8/9", "Adjusting learning rate"),
        "modify_batch_size": ("Step 8/9", "Adjusting batch size"),
        "modify_optimizer": ("Step 8/9", "Trying different optimizer"),
        "compare_results": ("Step 9/9", "Comparing results"),
    }

    total_reward = 0.0
    step = 0
    terminated = False
    truncated = False

    while not (terminated or truncated) and step < int(max_steps):
        action = agent.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)

        action_name = info.get("action_type", "unknown")
        label = step_labels.get(action_name, ("", action_name))
        total_reward += reward
        step += 1

        # Get latest logs from env
        latest_logs = info.get("logs", [])
        log_detail = latest_logs[-1] if latest_logs else ""

        phase_icon = {
            "parsing": "📄", "repo_analysis": "🔍", "setup": "📦",
            "execution": "🚀", "debugging": "🐛", "experimentation": "🧪",
            "comparison": "📊",
        }.get(info.get("phase", ""), "▶")

        metric_str = f" | metric: **{info.get('current_metric', 0):.3f}**" if info.get("current_metric", 0) > 0 else ""
        reward_str = f" | reward: {reward:+.2f}" if abs(reward) > 0.01 else ""

        line = f"{phase_icon} `{label[0]}` **{label[1]}**{metric_str}{reward_str}"
        if log_detail:
            line += f"\n  - {log_detail}"

        current_metrics = json.dumps({
            "step": step,
            "current_metric": round(info.get("current_metric", 0), 4),
            "target_metric": round(info.get("target_metric", 0), 4),
            "gap": round(info.get("gap", 0), 4),
            "total_reward": round(total_reward, 2),
            "phase": info.get("phase", ""),
            "success": info.get("success", False),
        }, indent=2)

        yield (log(line), paper_info_md, current_metrics, json.dumps(env.state.to_dict(), indent=2))
        time.sleep(0.15)

    # --- Final summary ---
    success = info.get("success", False)
    final_metric = info.get("current_metric", 0)
    gap = info.get("gap", 0)

    result_icon = "✅" if success else "⚠️"
    summary = f"""
---
### {result_icon} Reproduction {'Complete!' if success else 'Incomplete'}

| Metric | Value |
|--------|-------|
| Steps | {step} |
| Final Metric | {final_metric:.4f} |
| Target | {target_metric:.4f} |
| Gap | {gap:.4f} |
| Total Reward | {total_reward:.2f} |
| Success | {'Yes' if success else 'No'} |
"""
    if not success:
        summary += "\n*Try increasing max steps or enabling LLM for better results.*"

    yield (log(summary), paper_info_md,
           json.dumps({
               "final_metric": round(final_metric, 4),
               "target_metric": round(target_metric, 4),
               "gap": round(gap, 4),
               "steps": step,
               "total_reward": round(total_reward, 2),
               "success": success,
           }, indent=2),
           json.dumps(env.state.to_dict(), indent=2))


# ---------------------------------------------------------------------------
#  Tab 2: Simulation Demo (preserved from original)
# ---------------------------------------------------------------------------

class SimulationRunner:
    """Runs simulation episodes with pre-loaded papers."""

    def __init__(self):
        self.env = None
        self.agent = None

    def run_episode(
        self,
        difficulty: str,
        agent_type: str,
        max_steps: int,
        use_llm: bool,
        progress=gr.Progress(),
    ) -> Tuple[str, str, str, str]:
        try:
            self.env = ReproAgentEnv(
                difficulty=difficulty,
                max_steps=int(max_steps),
                use_llm=use_llm,
                render_mode=None,
            )
            self.agent = create_agent(self.env, agent_type=agent_type, use_llm=use_llm)

            obs, info = self.env.reset()
            self.agent.reset()

            progress(0, desc="Starting episode...")

            step = 0
            terminated = False
            truncated = False
            total_reward = 0.0
            step_logs: List[str] = []

            while not (terminated or truncated) and step < int(max_steps):
                progress((step + 1) / max_steps, desc=f"Step {step + 1}/{int(max_steps)}")

                action = self.agent.select_action(obs, info)
                reasoning = self.agent.get_reasoning(self.env.state, action)
                obs, reward, terminated, truncated, info = self.env.step(action)

                action_name = info.get("action_type", "unknown")
                latest = info.get("logs", [])
                log_line = latest[-1] if latest else ""

                step_log = (
                    f"### Step {step + 1}\n"
                    f"**Phase:** `{info.get('phase', '?')}`  \n"
                    f"**Action:** {action_name}  \n"
                    f"**Reasoning:** {reasoning}  \n"
                    f"**Reward:** {reward:.2f}  \n"
                    f"**Metric:** {info.get('current_metric', 0):.3f}\n"
                )
                if log_line:
                    step_log += f"\n> {log_line}\n"

                step_logs.append(step_log)
                total_reward += reward
                step += 1
                time.sleep(0.05)

            progress(1.0, desc="Done!")

            # Summary
            current_metric = info.get("current_metric", 0)
            target_metric = info.get("target_metric", 0)
            gap = info.get("gap", 0)
            success = terminated

            icon = "✅" if success else "❌"
            summary = f"""# {icon} Episode Summary

## Results

| Metric | Value |
|--------|-------|
| **Steps Taken** | {step} |
| **Total Reward** | {total_reward:.2f} |
| **Current Metric** | {current_metric:.3f} |
| **Target Metric** | {target_metric:.3f} |
| **Gap** | {gap:.3f} |
| **Success** | {'Yes' if success else 'No'} |

## Progress
Progress: {(current_metric / target_metric * 100) if target_metric > 0 else 0:.1f}%
"""
            if success:
                summary += "\n## 🎉 Reproduction Successful!"
            else:
                summary += f"\n## ⚠️ Reproduction Incomplete\nGap remaining: {gap:.3f}"

            metrics_json = json.dumps({
                "current_metric": current_metric,
                "target_metric": target_metric,
                "gap": gap,
                "success": success,
                "phase": info.get("phase", ""),
            }, indent=2)

            return (
                summary,
                "\n\n---\n\n".join(step_logs),
                metrics_json,
                json.dumps(self.env.state.to_dict(), indent=2),
            )

        except Exception as e:
            error_msg = f"**Error:** {e}\n\n```\n{traceback.format_exc()}\n```"
            return (error_msg, "", "{}", "{}")


# ---------------------------------------------------------------------------
#  Tab 3: Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(difficulty: str, num_episodes: int, progress=gr.Progress()):
    """Compare reasoning agent vs random agent."""
    try:
        results = {"reasoning": [], "random": []}

        for agent_type in ["reasoning", "random"]:
            for ep in range(int(num_episodes)):
                progress(
                    (ep + 1) / (int(num_episodes) * 2),
                    desc=f"{agent_type.title()} agent — episode {ep + 1}/{int(num_episodes)}",
                )

                env = ReproAgentEnv(difficulty=difficulty, max_steps=30, use_llm=False)
                agent = create_agent(env, agent_type=agent_type, use_llm=False)

                obs, info = env.reset()
                agent.reset()

                total_reward = 0
                steps = 0
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    action = agent.select_action(obs, info)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1

                results[agent_type].append({
                    "episode": ep + 1,
                    "success": terminated,
                    "steps": steps,
                    "reward": total_reward,
                    "metric": info.get("current_metric", 0),
                })

        progress(1.0, desc="Done!")

        # Build comparison markdown
        def stats(data):
            success_rate = sum(1 for d in data if d["success"]) / len(data) * 100
            avg_reward = np.mean([d["reward"] for d in data])
            avg_metric = np.mean([d["metric"] for d in data])
            avg_steps = np.mean([d["steps"] for d in data])
            return success_rate, avg_reward, avg_metric, avg_steps

        r_stats = stats(results["reasoning"])
        rand_stats = stats(results["random"])

        winner = "Reasoning Agent" if r_stats[0] >= rand_stats[0] else "Random Agent"

        report = f"""# Benchmark Results

**Difficulty:** {difficulty} | **Episodes per agent:** {int(num_episodes)}

| Metric | Reasoning Agent | Random Agent |
|--------|:-:|:-:|
| **Success Rate** | {r_stats[0]:.0f}% | {rand_stats[0]:.0f}% |
| **Avg Reward** | {r_stats[1]:.1f} | {rand_stats[1]:.1f} |
| **Avg Final Metric** | {r_stats[2]:.3f} | {rand_stats[2]:.3f} |
| **Avg Steps** | {r_stats[3]:.1f} | {rand_stats[3]:.1f} |

### Winner: **{winner}** 🏆
"""
        return report

    except Exception as e:
        return f"**Error:** {e}\n```\n{traceback.format_exc()}\n```"


# ---------------------------------------------------------------------------
#  Build Gradio App
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Dark premium theme overrides */
.gradio-container {
    max-width: 1200px !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
.header-block {
    text-align: center;
    padding: 28px 20px 18px;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #fff;
    border-radius: 14px;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}
.header-block h1 {
    margin: 0 0 4px 0;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-block p {
    margin: 4px 0 0;
    opacity: 0.85;
    font-size: 1.05rem;
}
.step-badge {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.85rem;
    color: #a78bfa;
    margin-right: 6px;
}
"""


def create_demo():
    """Create the full Gradio demo."""
    try:
        create_sample_papers()
    except Exception:
        pass

    sim_runner = SimulationRunner()

    with gr.Blocks(
        title="ReproAgent - ML Paper Reproduction",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.violet,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0f172a",
            body_background_fill_dark="#0f172a",
            block_background_fill="#1e293b",
            block_background_fill_dark="#1e293b",
            block_border_color="#334155",
            block_label_text_color="#94a3b8",
            block_title_text_color="#e2e8f0",
            input_background_fill="#0f172a",
            input_background_fill_dark="#0f172a",
            button_primary_background_fill="linear-gradient(135deg, #7c3aed 0%, #2563eb 100%)",
            button_primary_text_color="#ffffff",
        ),
    ) as demo:
        # --- Header ---
        gr.HTML("""
        <div class="header-block">
            <h1>ReproAgent</h1>
            <p>AI Agent for Reproducing ML Research Papers</p>
            <p style="font-size:0.85rem; opacity:0.6; margin-top:6px;">
                Upload a PDF &rarr; Agent reads paper &rarr; Finds repo &rarr; Runs code &rarr; Debugs errors &rarr; Tunes hyperparameters &rarr; Compares results
            </p>
        </div>
        """)

        with gr.Tabs():
            # ============================================================
            #  TAB 1 — Reproduce a Paper
            # ============================================================
            with gr.Tab("📄 Reproduce a Paper", id="tab_reproduce"):
                gr.Markdown("### Provide a paper to reproduce")

                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_upload = gr.File(
                            label="Upload PDF",
                            file_types=[".pdf"],
                            type="filepath",
                        )
                        paper_url = gr.Textbox(
                            label="Or paste paper / arXiv URL",
                            placeholder="https://arxiv.org/abs/2301.xxxxx  or  https://arxiv.org/pdf/2301.xxxxx.pdf",
                            lines=1,
                        )

                        gr.Markdown("---")

                        with gr.Row():
                            use_llm_tab1 = gr.Checkbox(
                                value=True,
                                label="Use LLM (Groq)",
                                info="Uses Groq API for intelligent parsing",
                            )
                            exec_mode = gr.Radio(
                                choices=["Simulation", "Real Execution"],
                                value="Simulation",
                                label="Execution Mode",
                                info="Simulation is faster & safer",
                            )

                        with gr.Row():
                            max_steps_tab1 = gr.Slider(
                                minimum=10, maximum=100, value=30, step=5,
                                label="Max Steps",
                            )
                            clone_dir_tab1 = gr.Textbox(
                                label="Clone Directory (for Real Execution)",
                                placeholder="/tmp/reproagent",
                                value="/tmp/reproagent",
                                lines=1,
                            )

                        reproduce_btn = gr.Button(
                            "🚀 Start Reproduction",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("📋 Agent Log"):
                                agent_log = gr.Markdown("*Upload a PDF or paste a URL to begin.*")

                            with gr.Tab("📄 Paper Info"):
                                paper_info_display = gr.Markdown("*Paper details will appear here.*")

                            with gr.Tab("📈 Metrics"):
                                metrics_display = gr.Code(language="json", label="Live Metrics")

                            with gr.Tab("🔍 State"):
                                state_display = gr.Code(language="json", label="Environment State")

                reproduce_btn.click(
                    fn=run_paper_reproduction,
                    inputs=[pdf_upload, paper_url, use_llm_tab1, max_steps_tab1, exec_mode, clone_dir_tab1],
                    outputs=[agent_log, paper_info_display, metrics_display, state_display],
                )

            # ============================================================
            #  TAB 2 — Simulation Demo
            # ============================================================
            with gr.Tab("🎮 Simulation Demo", id="tab_simulation"):
                gr.Markdown(
                    "### Quick simulation with pre-loaded papers\n"
                    "Test the agent on built-in paper configurations without uploading anything."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        sim_difficulty = gr.Radio(
                            ["easy", "medium", "hard"],
                            value="easy",
                            label="Difficulty",
                            info="Easy: Clean repo | Medium: Needs debugging | Hard: No code",
                        )
                        sim_agent = gr.Radio(
                            ["reasoning", "random"],
                            value="reasoning",
                            label="Agent Type",
                        )
                        sim_steps = gr.Slider(10, 100, value=30, step=5, label="Max Steps")
                        sim_llm = gr.Checkbox(value=False, label="Use LLM")
                        sim_btn = gr.Button("🚀 Run Simulation", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("📋 Summary"):
                                sim_summary = gr.Markdown("*Run a simulation to see results*")
                            with gr.Tab("📝 Step Log"):
                                sim_steplog = gr.Markdown("*Step logs appear here*")
                            with gr.Tab("📈 Metrics"):
                                sim_metrics = gr.Code(language="json", label="Metrics")
                            with gr.Tab("🔍 State"):
                                sim_state = gr.Code(language="json", label="State")

                sim_btn.click(
                    fn=sim_runner.run_episode,
                    inputs=[sim_difficulty, sim_agent, sim_steps, sim_llm],
                    outputs=[sim_summary, sim_steplog, sim_metrics, sim_state],
                )

            # ============================================================
            #  TAB 3 — Benchmark
            # ============================================================
            with gr.Tab("📊 Benchmark", id="tab_benchmark"):
                gr.Markdown(
                    "### Compare agents\n"
                    "Run multiple episodes and compare the Reasoning agent vs Random baseline."
                )

                with gr.Row():
                    bench_difficulty = gr.Radio(
                        ["easy", "medium", "hard"],
                        value="easy",
                        label="Difficulty",
                    )
                    bench_episodes = gr.Slider(
                        2, 20, value=5, step=1,
                        label="Episodes per agent",
                    )
                    bench_btn = gr.Button("📊 Run Benchmark", variant="primary")

                bench_result = gr.Markdown("*Click Run Benchmark to start*")

                bench_btn.click(
                    fn=run_benchmark,
                    inputs=[bench_difficulty, bench_episodes],
                    outputs=[bench_result],
                )

        # Footer
        gr.HTML("""
        <div style="text-align:center; padding:16px; opacity:0.5; font-size:0.8rem; margin-top:12px;">
            ReproAgent &mdash; AI Agent Hackathon 2024 &mdash;
            Gymnasium / OpenEnv compatible &mdash;
            Groq &bull; PyTorch &bull; Gradio
        </div>
        """)

    return demo


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
