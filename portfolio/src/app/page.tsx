import { Navbar } from "@/components/navbar";
import { Hero } from "@/components/hero";
import { Dashboard } from "@/components/dashboard";
import { AIWorkflows } from "@/components/ai-workflows";
import { ProjectGrid } from "@/components/project-grid";
import { ManagerCaseView } from "@/components/manager-case-view";
import { PrintButton } from "@/components/print-button";

export default function Home() {
  return (
    <>
      <Navbar />

      <main>
        <Hero />
        <Dashboard />
        <AIWorkflows />
        <ProjectGrid />
        <ManagerCaseView />
      </main>

      {/* Footer — visible in both modes */}
      <footer className="py-12 px-6 border-t border-slate-200 print-compact">
        <div className="max-w-6xl mx-auto text-center">
          <p className="text-sm text-slate-400 print:text-[8pt]">
            Executive Portfolio &middot; Built with Next.js, Tailwind CSS &amp;
            Framer Motion
          </p>
          <p className="hidden print-only print:block text-[8pt] text-slate-400 mt-1">
            This document was generated from an interactive web application.
            Visit the live version for the full experience.
          </p>
        </div>
      </footer>

      {/* Floating PDF button */}
      <PrintButton />
    </>
  );
}
