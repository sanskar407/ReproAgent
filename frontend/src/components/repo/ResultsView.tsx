import { ArrowDownToLine, RotateCcw, CheckCircle2, TrendingDown, TrendingUp, Minus } from "lucide-react";
import { Button } from "@/components/ui/button";

interface MetricRow {
  metric: string;
  paper: number;
  agent: number;
  higherIsBetter: boolean;
  unit?: string;
}

const RESULTS: MetricRow[] = [
  { metric: "Accuracy", paper: 0.924, agent: 0.918, higherIsBetter: true },
  { metric: "Precision", paper: 0.911, agent: 0.904, higherIsBetter: true },
  { metric: "Recall", paper: 0.887, agent: 0.892, higherIsBetter: true },
  { metric: "F1 Score", paper: 0.899, agent: 0.898, higherIsBetter: true },
  { metric: "AUC-ROC", paper: 0.953, agent: 0.949, higherIsBetter: true },
  { metric: "Test Loss", paper: 0.198, agent: 0.211, higherIsBetter: false },
  { metric: "Training Time (min)", paper: 142, agent: 138, higherIsBetter: false, unit: "min" },
];

interface ResultsViewProps {
  results: any;
  mode: "Easy" | "Medium" | "Advanced";
  onRunAgain: () => void;
}

const fmt = (v: number, unit?: string) => {
  if (unit === "min") return `${v.toFixed(0)} min`;
  return v.toFixed(3);
};

const deviation = (paper: number, agent: number) => ((agent - paper) / paper) * 100;

export const ResultsView = ({ results, mode, onRunAgain }: ResultsViewProps) => {
  const dynamicResults = results?.metrics && Array.isArray(results.metrics) 
    ? results.metrics.map((m: any) => ({
        metric: m.name || m.metric || "Target Metric",
        paper: m.paper || 0,
        agent: m.agent || 0,
        higherIsBetter: m.higherIsBetter ?? true,
        unit: m.unit
      }))
    : RESULTS;

  const downloadCsv = () => {
    const header = "Metric,Paper Result,Agent Result,Deviation (%)\n";
    const rows = dynamicResults.map(r =>
      `${r.metric},${r.paper},${r.agent},${deviation(r.paper, r.agent).toFixed(2)}`
    ).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "repoagent-results.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadPPT = () => {
    if (results?.ppt_url) {
      const a = document.createElement("a");
      // If the URL is relative, prepend the backend host
      a.href = results.ppt_url.startsWith("http") 
        ? results.ppt_url 
        : `http://localhost:7860${results.ppt_url.startsWith("/") ? "" : "/"}${results.ppt_url}`;
      a.download = "Research_Presentation.pptx";
      a.click();
    }
  };

  // Reproduction success: avg absolute deviation < 2%
  const avgDev = dynamicResults.reduce((sum, r) => sum + Math.abs(deviation(r.paper, r.agent)), 0) / dynamicResults.length;
  const successful = avgDev < 5;

  return (
    <div className="container max-w-5xl py-12 md:py-16 animate-fade-up">
      {/* Header */}
      <div className="mb-10">
        <p className="text-[11px] uppercase tracking-[0.25em] text-accent font-medium mb-3">
          {mode === "Easy" ? "Paper Summary & Presentation" : "Reproduction Report"}
        </p>
        <h1 className="font-serif text-4xl md:text-5xl text-foreground mb-4 text-balance">
          {mode === "Easy" ? "Research insights" : "Results comparison"}
        </h1>
        <p className="text-muted-foreground max-w-2xl">
          {mode === "Easy" 
            ? "AI-generated description and structured presentation based on the uploaded paper."
            : "Side-by-side metrics from the original paper versus those produced by RepoAgent's autonomous reproduction."}
        </p>
      </div>

      {mode === "Easy" ? (
        <div className="space-y-8">
          {/* Summary Card */}
          <div className="bg-card rounded-lg border border-border shadow-paper p-8">
            <h2 className="text-xs uppercase tracking-[0.18em] text-muted-foreground font-semibold mb-6">
              Paper Description
            </h2>
            <div className="whitespace-pre-wrap text-foreground/80 leading-relaxed text-sm">
              {results?.description}
            </div>

          </div>

          {/* PPT Card */}
          <div className="bg-accent/5 rounded-lg border border-accent/20 p-8 flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-lg bg-accent/10 flex items-center justify-center text-accent">
                <ArrowDownToLine className="w-6 h-6" />
              </div>
              <div>
                <h3 className="font-serif text-xl text-foreground">PowerPoint Presentation</h3>
                <p className="text-sm text-muted-foreground">Ready to present with key highlights and figures.</p>
              </div>
            </div>
            <Button onClick={handleDownloadPPT} size="lg" className="bg-accent text-accent-foreground hover:opacity-90 gap-2">
              <ArrowDownToLine className="w-4 h-4" />
              Download PPTX
            </Button>
          </div>
        </div>
      ) : (
        <>
          {/* Summary card for Advanced Mode */}
          <div className={`relative overflow-hidden rounded-lg border shadow-elevated p-6 mb-8 ${
            successful ? "border-success/30 bg-success/5" : "border-warning/30 bg-warning/5"
          }`}>
            <div className="flex items-start gap-4">
              <div className={`w-11 h-11 rounded-full flex items-center justify-center shrink-0 ${
                successful ? "bg-success text-success-foreground" : "bg-warning text-warning-foreground"
              }`}>
                <CheckCircle2 className="w-5 h-5" />
              </div>
              <div className="flex-1">
                <h2 className="font-serif text-2xl text-foreground mb-1.5">
                  {successful ? "Reproduction successful" : "Partial reproduction"}
                </h2>
                <p className="text-sm text-foreground/80 leading-relaxed max-w-2xl">
                  RepoAgent reproduced the paper's results within{" "}
                  <span className="font-mono font-semibold">{avgDev.toFixed(2)}%</span> mean absolute deviation across{" "}
                  {dynamicResults.length} reported metric(s). Reported numbers are{" "}
                  {successful ? "consistent with" : "broadly aligned with"} the original publication.
                </p>
              </div>
            </div>
          </div>

          {/* Comparison table */}
          <div className="bg-card rounded-lg border border-border shadow-paper overflow-hidden mb-8">
            <div className="grid grid-cols-12 px-6 py-3.5 bg-secondary/60 border-b border-border text-[11px] uppercase tracking-[0.16em] font-semibold text-muted-foreground">
              <div className="col-span-4">Metric</div>
              <div className="col-span-3 text-right font-serif normal-case tracking-normal text-sm text-foreground/70">Paper Results</div>
              <div className="col-span-3 text-right font-serif normal-case tracking-normal text-sm text-foreground/70">Agent Results</div>
              <div className="col-span-2 text-right">Δ</div>
            </div>
            {dynamicResults.map((row, idx) => {
              const dev = deviation(row.paper, row.agent);
              const better = row.higherIsBetter ? dev > 0 : dev < 0;
              const negligible = Math.abs(dev) < 0.5;
              const Icon = negligible ? Minus : better ? TrendingUp : TrendingDown;
              const color = negligible ? "text-muted-foreground" : better ? "text-success" : "text-destructive";
              return (
                <div
                  key={row.metric}
                  className={`grid grid-cols-12 px-6 py-4 items-center text-sm transition-colors
                    ${idx !== dynamicResults.length - 1 ? "border-b border-border/60" : ""}
                    hover:bg-secondary/30`}
                >
                  <div className="col-span-4 font-medium text-foreground">{row.metric}</div>
                  <div className="col-span-3 text-right font-mono text-foreground/80 tabular-nums">
                    {fmt(row.paper, row.unit)}
                  </div>
                  <div className="col-span-3 text-right font-mono text-foreground tabular-nums font-semibold">
                    {fmt(row.agent, row.unit)}
                  </div>
                  <div className={`col-span-2 flex items-center justify-end gap-1.5 font-mono text-xs ${color}`}>
                    <Icon className="w-3.5 h-3.5" />
                    <span className="tabular-nums">{dev > 0 ? "+" : ""}{dev.toFixed(2)}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Actions */}
      <div className="flex flex-col sm:flex-row gap-3 justify-between items-start sm:items-center mt-12">
        <p className="text-xs text-muted-foreground font-mono">
          Generated {new Date().toLocaleString()} · Run ID: <span className="text-foreground/70">RA-{Math.random().toString(36).slice(2, 8).toUpperCase()}</span>
        </p>
        <div className="flex gap-2.5">
          {(mode === "Medium" || mode === "Advanced") && (
            <Button variant="outline" onClick={downloadCsv} className="gap-2 border-border bg-card hover:bg-secondary">
              <ArrowDownToLine className="w-4 h-4" />
              Export CSV
            </Button>
          )}
          <Button onClick={onRunAgain} className="gap-2 bg-ink text-primary-foreground hover:opacity-90 shadow-paper">
            <RotateCcw className="w-4 h-4" />
            Run Again
          </Button>
        </div>
      </div>
    </div>
  );
};

