import { useRef, useState } from "react";
import { z } from "zod";
import { Upload, Link2, FileText, X, ArrowRight, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const urlSchema = z
  .string()
  .trim()
  .max(500)
  .url({ message: "Please enter a valid URL" })
  .optional()
  .or(z.literal(""));

interface LandingFormProps {
  onSubmit: (payload: { 
    file: File | null; 
    url: string; 
    mode: "Easy" | "Medium" | "Advanced";
    useLLM: boolean;
    execMode: string;
    maxSteps: number;
    cloneDir: string;
  }) => void;
}

export const LandingForm = ({ onSubmit }: LandingFormProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File | null) => {
    setError(null);
    if (!f) return setFile(null);
    if (f.type !== "application/pdf") {
      setError("Only PDF files are supported");
      return;
    }
    if (f.size > 20 * 1024 * 1024) {
      setError("File must be under 20MB");
      return;
    }
    setFile(f);
  };

  const [mode, setMode] = useState<"Easy" | "Medium" | "Advanced">("Advanced");
  const [useLLM, setUseLLM] = useState(true);
  const [execMode, setExecMode] = useState("Simulation");
  const [maxSteps, setMaxSteps] = useState(30);
  const [cloneDir, setCloneDir] = useState("/tmp/reproagent");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!file && !url.trim()) {
      setError("Provide a PDF or paper URL to begin");
      return;
    }

    if (url.trim()) {
      const result = urlSchema.safeParse(url);
      if (!result.success) {
        setError(result.error.issues[0]?.message ?? "Invalid URL");
        return;
      }
    }

    toast.success(`Paper accepted — initiating ${mode} pipeline`);
    onSubmit({ 
      file, 
      url: url.trim(), 
      mode,
      useLLM,
      execMode,
      maxSteps,
      cloneDir
    });
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto space-y-5">
      {/* Mode Selector */}
      <div className="flex justify-center mb-6">
        <div className="inline-flex p-1 bg-secondary rounded-lg border border-border shadow-paper">
          <button
            type="button"
            onClick={() => setMode("Advanced")}
            className={`px-4 py-1.5 rounded-md text-[10px] font-medium transition-smooth ${
              mode === "Advanced"
                ? "bg-ink text-primary-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Advanced
          </button>
          <button
            type="button"
            onClick={() => setMode("Medium")}
            className={`px-4 py-1.5 rounded-md text-[10px] font-medium transition-smooth ${
              mode === "Medium"
                ? "bg-warning text-warning-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Medium
          </button>
          <button
            type="button"
            onClick={() => setMode("Easy")}
            className={`px-4 py-1.5 rounded-md text-[10px] font-medium transition-smooth ${
              mode === "Easy"
                ? "bg-accent text-accent-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Easy
          </button>
        </div>
      </div>


      {/* Upload zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFile(e.dataTransfer.files[0] ?? null);
        }}
        onClick={() => inputRef.current?.click()}
        className={`relative cursor-pointer rounded-lg border border-dashed p-8 bg-card transition-smooth shadow-paper
          ${dragOver ? "border-accent bg-accent/5" : "border-border hover:border-foreground/30 hover:bg-secondary/40"}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded bg-secondary flex items-center justify-center shrink-0">
                <FileText className="w-5 h-5 text-accent" />
              </div>
              <div className="min-w-0">
                <p className="text-sm font-medium truncate">{file.name}</p>
                <p className="text-xs text-muted-foreground">
                  {(file.size / 1024).toFixed(1)} KB · PDF
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); setFile(null); }}
              className="p-1.5 rounded hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Remove file"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center gap-3">
            <div className="w-12 h-12 rounded-full bg-secondary flex items-center justify-center">
              <Upload className="w-5 h-5 text-foreground/70" />
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">
                Drop your paper here, or <span className="text-accent">browse</span>
              </p>
              <p className="text-xs text-muted-foreground mt-1">PDF up to 20MB</p>
            </div>
          </div>
        )}
      </div>

      {/* Divider */}
      <div className="flex items-center gap-3 py-1">
        <div className="flex-1 h-px bg-border" />
        <span className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground font-medium">or</span>
        <div className="flex-1 h-px bg-border" />
      </div>

      {/* URL input */}
      <div className="relative">
        <Link2 className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://arxiv.org/abs/2401.00001"
          maxLength={500}
          className="w-full h-12 pl-10 pr-4 rounded-lg bg-card border border-border shadow-paper
            text-sm placeholder:text-muted-foreground/70 font-mono
            focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent transition-smooth"
        />
      </div>

      {/* Advanced Controls (Only for Medium/Advanced) */}
      {(mode === "Medium" || mode === "Advanced") && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 rounded-lg bg-card border border-border shadow-paper animate-fade-up">
          <div className="space-y-2">
            <label className="text-[11px] uppercase tracking-wider font-semibold text-muted-foreground">
              Intelligence
            </label>
            <div className="flex items-center space-x-2 pt-1">
              <input 
                type="checkbox" 
                id="useLLM" 
                checked={useLLM} 
                onChange={(e) => setUseLLM(e.target.checked)}
                className="w-4 h-4 rounded border-border text-accent focus:ring-accent"
              />
              <label htmlFor="useLLM" className="text-sm text-foreground/80">
                Uses Groq API for intelligent parsing
              </label>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-[11px] uppercase tracking-wider font-semibold text-muted-foreground">
              Execution Mode
            </label>
            <div className="flex space-x-4 pt-1">
              {["Simulation", "Real Execution"].map((m) => (
                <label key={m} className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="radio"
                    name="execMode"
                    value={m}
                    checked={execMode === m}
                    onChange={(e) => setExecMode(e.target.value)}
                    className="w-4 h-4 text-accent border-border focus:ring-accent"
                  />
                  <span className="text-sm text-foreground/80">{m}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <label className="text-[11px] uppercase tracking-wider font-semibold text-muted-foreground">
                Max Steps
              </label>
              <span className="text-xs font-mono text-accent">{maxSteps}</span>
            </div>
            <input
              type="range"
              min="10"
              max="100"
              step="5"
              value={maxSteps}
              onChange={(e) => setMaxSteps(parseInt(e.target.value))}
              className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-accent"
            />
          </div>

          <div className="space-y-2">
            <label className="text-[11px] uppercase tracking-wider font-semibold text-muted-foreground">
              Clone Directory
            </label>
            <input
              type="text"
              value={cloneDir}
              onChange={(e) => setCloneDir(e.target.value)}
              placeholder="/tmp/reproagent"
              className="w-full h-9 px-3 rounded border border-border bg-background text-sm focus:outline-none focus:ring-1 focus:ring-accent"
            />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <p className="text-sm text-destructive font-medium animate-fade-in">{error}</p>
      )}

      {/* Submit */}
      <Button
        type="submit"
        size="lg"
        className="w-full h-12 bg-ink text-primary-foreground hover:opacity-90 shadow-elevated
          group font-medium tracking-wide transition-smooth"
      >
        <Sparkles className="w-4 h-4 mr-2 opacity-80" />
        {mode === "Easy" ? "Generate Summary & PPT" : "Reproduce This Paper"}
        <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-0.5 transition-transform" />
      </Button>

      <p className="text-xs text-muted-foreground text-center">
        {mode === "Easy" 
          ? "RepoAgent will summarize the paper and create an impressive presentation for you."
          : "RepoAgent will locate the source repository, replicate the environment, and re-run experiments."}
      </p>
    </form>

  );
};
