"use client";

import { aiWorkflows } from "@/lib/portfolio-data";
import { AIWorkflowCard } from "./ai-workflow-card";
import { SectionHeader } from "./section-header";

export function AIWorkflows() {
  return (
    <section id="workflows" className="py-24 px-6 print-compact print-page-break">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          label="AI Development"
          title="Workflow Architecture & Delivery"
          description="End-to-end AI pipelines I've designed and deployed — from problem identification through production deployment and measurable ROI."
        />

        <div className="space-y-8">
          {aiWorkflows.map((workflow, index) => (
            <AIWorkflowCard
              key={workflow.id}
              workflow={workflow}
              index={index}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
