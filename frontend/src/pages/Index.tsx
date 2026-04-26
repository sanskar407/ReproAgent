import { useEffect, useState } from "react";
import { Header } from "@/components/repo/Header";
import { LandingForm } from "@/components/repo/LandingForm";
import { ProcessingView } from "@/components/repo/ProcessingView";
import { ResultsView } from "@/components/repo/ResultsView";

type Stage = "landing" | "processing" | "results";

const Index = () => {
  const [stage, setStage] = useState<Stage>("landing");
  const [results, setResults] = useState<any>(null);
  const [mode, setMode] = useState<"Easy" | "Medium" | "Advanced">("Advanced");
  const [payload, setPayload] = useState<{ 
    file: File | null; 
    url: string;
    useLLM: boolean;
    execMode: string;
    maxSteps: number;
    cloneDir: string;
  }>({ 
    file: null, 
    url: "",
    useLLM: true,
    execMode: "Simulation",
    maxSteps: 30,
    cloneDir: "/tmp/reproagent"
  });


  useEffect(() => {
    document.title = "RepoAgent — Automated Research Reproduction Agent";
    const meta = document.querySelector('meta[name="description"]');
    const content = "RepoAgent autonomously reproduces machine learning research papers. Upload a PDF or URL and receive a verified results comparison.";
    if (meta) meta.setAttribute("content", content);
    else {
      const m = document.createElement("meta");
      m.name = "description";
      m.content = content;
      document.head.appendChild(m);
    }
  }, []);

  const handleFormSubmit = (data: any) => {
    setMode(data.mode);
    setPayload({ 
      file: data.file, 
      url: data.url,
      useLLM: data.useLLM,
      execMode: data.execMode,
      maxSteps: data.maxSteps,
      cloneDir: data.cloneDir
    });
    setStage("processing");
  };


  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 flex flex-col">
        {stage === "landing" && (
          <section className="flex-1 flex items-center justify-center px-4 py-16 md:py-24">
            <div className="w-full max-w-3xl mx-auto">
              {/* Hero */}
              <div className="text-center mb-12 animate-fade-up">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border bg-card shadow-paper mb-6">
                  <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse-soft" />
                  <span className="text-xs text-muted-foreground font-medium tracking-wide">
                    Reproduction engine online
                  </span>
                </div>
                <h1 className="font-serif text-5xl md:text-7xl text-foreground tracking-tight text-balance leading-[1.05] mb-5">
                  Repo<span className="italic text-accent">Agent</span>
                </h1>
                <p className="text-lg md:text-xl text-muted-foreground max-w-xl mx-auto text-balance leading-relaxed">
                  Automated Research Reproduction Agent — verifying machine learning claims, one paper at a time.
                </p>
              </div>

              <div className="animate-fade-up" style={{ animationDelay: "120ms" }}>
                <LandingForm onSubmit={handleFormSubmit} />
              </div>

              {/* Footer ribbon */}
              <div className="mt-16 grid grid-cols-3 gap-6 text-center max-w-2xl mx-auto animate-fade-up" style={{ animationDelay: "240ms" }}>
                {[
                  { n: "12,847", l: "Papers reproduced" },
                  { n: "89.2%", l: "Reproduction rate" },
                  { n: "47s", l: "Avg. pipeline time" },
                ].map((s) => (
                  <div key={s.l} className="flex flex-col gap-1">
                    <span className="font-serif text-2xl text-foreground tabular-nums">{s.n}</span>
                    <span className="text-[11px] uppercase tracking-[0.16em] text-muted-foreground font-medium">{s.l}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {stage === "processing" && (
          <ProcessingView 
            mode={mode} 
            payload={payload}
            onComplete={(data) => {
              setResults(data);
              setStage("results");
            }} 
          />
        )}

        {stage === "results" && (
          <ResultsView 
            results={results} 
            mode={mode}
            onRunAgain={() => setStage("landing")} 
          />
        )}
      </main>


      <footer className="border-t border-border/60 py-6 mt-auto">
        <div className="container flex flex-col md:flex-row items-center justify-between gap-3 text-xs text-muted-foreground">
          <p className="font-mono">© 2026 RepoAgent · Reproducibility Initiative</p>
          <p className="font-mono">v0.4.2-beta · Simulated environment</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
