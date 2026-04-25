import { FlaskConical } from "lucide-react";

export const Header = () => {
  return (
    <header className="w-full border-b border-border/60 backdrop-blur-sm bg-background/70 sticky top-0 z-40">
      <div className="container flex items-center justify-between h-16">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-md bg-ink flex items-center justify-center shadow-paper">
            <FlaskConical className="w-4 h-4 text-primary-foreground" strokeWidth={2} />
          </div>
          <div className="flex flex-col leading-none">
            <span className="font-serif text-lg text-foreground">RepoAgent</span>
            <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground font-medium">
              Reproduction Lab
            </span>
          </div>
        </div>
        <nav className="hidden md:flex items-center gap-7 text-sm text-muted-foreground">
          <a href="#" className="hover:text-foreground transition-colors">Documentation</a>
          <a href="#" className="hover:text-foreground transition-colors">Benchmarks</a>
          <a href="#" className="hover:text-foreground transition-colors">About</a>
        </nav>
      </div>
    </header>
  );
};
