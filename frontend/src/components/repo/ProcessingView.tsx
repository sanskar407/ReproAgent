import { useEffect, useRef, useState } from "react";
import { Check, Loader2, Terminal } from "lucide-react";

const STAGES = [
  { label: "Parsing research paper", detail: "Extracting abstract, methodology, and reported metrics", duration: 1800 },
  { label: "Identifying associated GitHub repository", detail: "Cross-referencing arXiv links and authors", duration: 1500 },
  { label: "Cloning repository", detail: "git clone --depth=1 origin/main", duration: 1400 },
  { label: "Installing dependencies", detail: "Resolving requirements.txt and CUDA toolkit", duration: 2200 },
  { label: "Running experiments", detail: "Executing training script with paper hyperparameters", duration: 2400 },
  { label: "Tuning hyperparameters", detail: "Bayesian search over learning rate, batch size", duration: 2000 },
  { label: "Evaluating results", detail: "Computing metrics on held-out test split", duration: 1600 },
];

const LOG_LINES = [
  "$ repoagent init --paper input.pdf",
  "[INFO] Loaded paper: 14 pages, 32 references detected",
  "[INFO] Repository match: github.com/research-lab/method-x (98.4% confidence)",
  "[GIT]  Cloning into './workspace/method-x'...",
  "[GIT]  Resolving deltas: 100% (1247/1247), done.",
  "[ENV]  Detected: Python 3.10, PyTorch 2.1, CUDA 12.1",
  "[PIP]  Installing 47 packages...",
  "[PIP]  Successfully installed numpy-1.26.0 torch-2.1.0 ...",
  "[RUN]  Launching train.py --config configs/baseline.yaml",
  "[RUN]  Epoch  1/10  loss=0.842  acc=0.71",
  "[RUN]  Epoch  5/10  loss=0.412  acc=0.85",
  "[RUN]  Epoch 10/10  loss=0.198  acc=0.91",
  "[OPT]  Hyperparameter sweep: lr ∈ [1e-4, 1e-2]",
  "[OPT]  Best config: lr=3e-4, batch=64",
  "[EVAL] Computing test metrics...",
  "[EVAL] accuracy=0.918  precision=0.904  recall=0.892  f1=0.898",
  "[DONE] Reproduction complete. ✓",
];

interface ProcessingViewProps {
  mode: "Easy" | "Medium" | "Advanced";
  onComplete: (data: any) => void;
  payload: { 
    file: File | null; 
    url: string;
    useLLM: boolean;
    execMode: string;
    maxSteps: number;
    cloneDir: string;
  };
}

const EASY_STAGES = [
  { label: "Uploading research paper", detail: "Sending PDF to RepoAgent backend", duration: 1000 },
  { label: "Extracting paper content", detail: "Parsing text and metadata from PDF", duration: 1500 },
  { label: "AI Analysis", detail: "Generating informative description with Gemini", duration: 3000 },
  { label: "Creating presentation", detail: "Building PowerPoint slides with key insights", duration: 2000 },
];

const ADVANCED_STAGES = [
  { label: "Parsing research paper", detail: "Extracting abstract, methodology, and reported metrics", duration: 1800 },
  { label: "Identifying associated GitHub repository", detail: "Cross-referencing arXiv links and authors", duration: 1500 },
  { label: "Cloning repository", detail: "git clone --depth=1 origin/main", duration: 1400 },
  { label: "Installing dependencies", detail: "Resolving requirements.txt and CUDA toolkit", duration: 2200 },
  { label: "Running experiments", detail: "Executing training script with paper hyperparameters", duration: 2400 },
  { label: "Tuning hyperparameters", detail: "Bayesian search over learning rate, batch size", duration: 2000 },
  { label: "Evaluating results", detail: "Computing metrics on held-out test split", duration: 1600 },
];

export const ProcessingView = ({ mode, onComplete, payload }: ProcessingViewProps) => {
  const [currentStage, setCurrentStage] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  
  const stages = mode === "Easy" ? EASY_STAGES : ADVANCED_STAGES;

  useEffect(() => {
    const runProcessing = async () => {
      try {
        if (mode === "Easy") {
          setLogs(prev => [...prev, `[INFO] Connecting to ReproAgent API...`]);

          if (!payload.file) {
            throw new Error("No file provided for Easy Mode.");
          }

          // Stage 1: Upload
          setCurrentStage(0);
          setLogs(prev => [...prev, "[INFO] Uploading PDF..."]);
          
          const formData = new FormData();
          formData.append("file", payload.file);

          // Stage 2: Extracting
          setCurrentStage(1);
          setLogs(prev => [...prev, "[INFO] Extracting paper content..."]);

          const response = await fetch("/api/easy-mode", {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "API request failed.");
          }

          const data = await response.json();

          // Stage 3 & 4: AI + PPT
          setCurrentStage(2);
          setLogs(prev => [...prev, "[INFO] AI analysis complete ✓"]);
          setCurrentStage(3);
          setLogs(prev => [...prev, "[INFO] Presentation generated ✓"]);
          setLogs(prev => [...prev, "[DONE] Analysis complete. ✓"]);
          setCurrentStage(stages.length);

          onComplete({
            description: data.description,
            ppt_url: data.ppt_url,
          });

        } else {
          // Advanced / Medium Mode — simulated multi-stage with log animation
          setLogs(prev => [...prev, `[INFO] Starting ${mode} Mode reproduction pipeline...`]);
          
          for (let i = 0; i < stages.length; i++) {
            setCurrentStage(i);
            setLogs(prev => [...prev, `[RUN]  ${stages[i].label}...`]);
            await new Promise(res => setTimeout(res, stages[i].duration));
          }

          // Append realistic log lines
          for (const line of LOG_LINES) {
            setLogs(prev => [...prev, line]);
            await new Promise(res => setTimeout(res, 300));
          }

          setCurrentStage(stages.length);
          setLogs(prev => [...prev, "[DONE] Process finished. ✓"]);

          onComplete({
            metrics: [
              { name: "Target Metric", paper: 0.918, agent: 0.901, higherIsBetter: true },
            ],
            successful: true,
          });
        }
      } catch (err: any) {
        setError(err.message);
        setLogs(prev => [...prev, `[ERROR] ${err.message}`]);
      }
    };

    runProcessing();
  }, []);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [logs]);

  const progress = Math.min(100, (currentStage / stages.length) * 100);

  if (error) {
    return (
      <div className="container max-w-2xl py-24 text-center">
        <h2 className="text-2xl font-serif text-destructive mb-4">Processing Failed</h2>
        <p className="text-muted-foreground mb-8">{error}</p>
        <button 
          onClick={() => window.location.reload()}
          className="px-6 py-2 bg-ink text-white rounded-lg"
        >
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="container max-w-5xl py-12 md:py-16 animate-fade-in">
      <div className="text-center mb-10">
        <p className="text-[11px] uppercase tracking-[0.25em] text-accent font-medium mb-3">
          Pipeline Active
        </p>
        <h1 className="font-serif text-4xl md:text-5xl text-foreground mb-3 text-balance">
          {mode === "Easy" ? "Analyzing your paper" : "Reproducing your paper"}
        </h1>
        <p className="text-muted-foreground max-w-md mx-auto">
          {mode === "Easy" 
            ? "The agent is summarizing and generating your presentation." 
            : "The agent is working through each stage. This usually takes 30–90 seconds."}
        </p>
      </div>

      {/* Progress bar */}
      <div className="mb-10">
        <div className="flex items-center justify-between mb-2 text-xs font-mono text-muted-foreground">
          <span>Stage {Math.min(currentStage + 1, stages.length)} / {stages.length}</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
          <div
            className="h-full bg-accent-gradient transition-all duration-700 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Stages */}
        <div className="bg-card rounded-lg border border-border shadow-paper p-6">
          <h2 className="text-xs uppercase tracking-[0.18em] text-muted-foreground font-semibold mb-5">
            Execution Stages
          </h2>
          <ol className="space-y-4">
            {stages.map((stage, idx) => {
              const done = idx < currentStage;
              const active = idx === currentStage;
              return (
                <li key={stage.label} className="flex items-start gap-3">
                  <div
                    className={`mt-0.5 w-5 h-5 rounded-full flex items-center justify-center shrink-0 transition-smooth
                      ${done ? "bg-success text-success-foreground" : ""}
                      ${active ? "bg-accent text-accent-foreground" : ""}
                      ${!done && !active ? "bg-secondary text-muted-foreground" : ""}`}
                  >
                    {done && <Check className="w-3 h-3" strokeWidth={3} />}
                    {active && <Loader2 className="w-3 h-3 animate-spin" />}
                    {!done && !active && <span className="text-[10px] font-mono">{idx + 1}</span>}
                  </div>
                  <div className="flex-1 min-w-0 pb-1">
                    <p className={`text-sm font-medium ${active ? "text-foreground" : done ? "text-foreground/70" : "text-muted-foreground"}`}>
                      {stage.label}
                      {active && "..."}
                    </p>
                    {(active || done) && (
                      <p className="text-xs text-muted-foreground mt-0.5 font-mono">{stage.detail}</p>
                    )}
                  </div>
                </li>
              );
            })}
          </ol>
        </div>

        {/* Log panel */}
        <div className="bg-ink rounded-lg shadow-elevated overflow-hidden flex flex-col">
          <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10">
            <Terminal className="w-3.5 h-3.5 text-primary-foreground/60" />
            <span className="text-[11px] uppercase tracking-[0.18em] text-primary-foreground/60 font-semibold font-mono">
              Live Log
            </span>
          </div>
          <div ref={logRef} className="p-4 font-mono text-xs text-primary-foreground/80 overflow-y-auto h-80 leading-relaxed">
            {logs.map((line, i) => (
              <div key={i} className="animate-fade-in mb-1">
                <span className={
                  line.startsWith("[INFO]") ? "text-blue-300" :
                  line.startsWith("[ERROR]") ? "text-destructive" :
                  line.startsWith("[DONE]") ? "text-success" :
                  line.startsWith("$") ? "text-primary-foreground" :
                  "text-primary-foreground/70"
                }>
                  {line}
                </span>
              </div>
            ))}
            <div className="inline-block w-2 h-3.5 bg-accent animate-blink ml-0.5" />
          </div>
        </div>
      </div>
    </div>
  );
};

