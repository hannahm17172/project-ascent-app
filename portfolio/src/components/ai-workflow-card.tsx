"use client";

import { motion } from "framer-motion";
import { ChevronRight } from "lucide-react";
import type { AIWorkflow } from "@/lib/portfolio-data";
import { IconResolver } from "./icon-resolver";

interface AIWorkflowCardProps {
  workflow: AIWorkflow;
  index: number;
}

const stepColors = {
  problem: {
    border: "border-t-red-500",
    bg: "bg-red-50",
    text: "text-red-700",
    icon: "text-red-500",
    printLabel: "PROBLEM",
  },
  agent: {
    border: "border-t-blue-500",
    bg: "bg-blue-50",
    text: "text-blue-700",
    icon: "text-blue-500",
    printLabel: "AI AGENT",
  },
  process: {
    border: "border-t-amber-500",
    bg: "bg-amber-50",
    text: "text-amber-700",
    icon: "text-amber-500",
    printLabel: "PROCESS",
  },
  outcome: {
    border: "border-t-emerald-500",
    bg: "bg-emerald-50",
    text: "text-emerald-700",
    icon: "text-emerald-500",
    printLabel: "OUTCOME",
  },
};

type StepType = keyof typeof stepColors;

interface StepNodeProps {
  type: StepType;
  label: string;
  description: string;
  icon: string;
  extra?: React.ReactNode;
  isLast?: boolean;
}

function StepNode({
  type,
  label,
  description,
  icon,
  extra,
  isLast,
}: StepNodeProps) {
  const colors = stepColors[type];

  return (
    <>
      <div
        className={`flex-1 min-w-0 rounded-lg border ${colors.border} border-t-[3px] border-slate-200 bg-white p-5 print-flow-step print-avoid-break`}
      >
        {/* Step type label */}
        <div className="flex items-center gap-2 mb-3">
          <div
            className={`w-8 h-8 rounded-md ${colors.bg} flex items-center justify-center print:bg-transparent`}
          >
            <IconResolver name={icon} size={16} className={colors.icon} />
          </div>
          <span
            className={`text-xs font-bold uppercase tracking-wider ${colors.text}`}
          >
            {colors.printLabel}
          </span>
        </div>

        {/* Label */}
        <h4 className="font-semibold text-slate-900 text-sm mb-1 print:text-[9pt]">
          {label}
        </h4>

        {/* Description */}
        <p className="text-xs text-slate-500 leading-relaxed print:text-[8pt]">
          {description}
        </p>

        {/* Extra content (model name, steps, metric) */}
        {extra && <div className="mt-3">{extra}</div>}
      </div>

      {/* Arrow connector — hidden on print & on last item */}
      {!isLast && (
        <div className="flex items-center justify-center w-8 shrink-0 no-print">
          <ChevronRight size={20} className="text-slate-300" />
        </div>
      )}
    </>
  );
}

export function AIWorkflowCard({ workflow, index }: AIWorkflowCardProps) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden print-workflow-card print-avoid-break"
    >
      {/* Card Header */}
      <div className="px-6 pt-6 pb-4 border-b border-slate-100">
        <h3 className="text-xl font-bold text-slate-900 print:text-[12pt]">
          {workflow.title}
        </h3>
        <p className="mt-1 text-sm text-slate-500 leading-relaxed print:text-[9pt]">
          {workflow.subtitle}
        </p>
      </div>

      {/* Flow Steps */}
      <div className="px-6 py-6 print-flow-steps">
        <div className="flex items-stretch gap-0 overflow-x-auto print:overflow-visible">
          <StepNode
            type="problem"
            label={workflow.problem.label}
            description={workflow.problem.description}
            icon={workflow.problem.icon}
          />
          <StepNode
            type="agent"
            label={workflow.agent.label}
            description={workflow.agent.description}
            icon={workflow.agent.icon}
            extra={
              <span className="inline-block text-[11px] font-mono bg-slate-100 text-slate-600 px-2 py-0.5 rounded print:bg-transparent print:text-[7pt]">
                {workflow.agent.model}
              </span>
            }
          />
          <StepNode
            type="process"
            label={workflow.process.label}
            description={workflow.process.description}
            icon={workflow.process.icon}
            extra={
              <ul className="space-y-1">
                {workflow.process.steps.map((step, i) => (
                  <li
                    key={i}
                    className="text-[11px] text-slate-500 flex items-start gap-1.5 print:text-[7pt]"
                  >
                    <span className="w-1 h-1 rounded-full bg-amber-400 mt-1.5 shrink-0" />
                    {step}
                  </li>
                ))}
              </ul>
            }
          />
          <StepNode
            type="outcome"
            label={workflow.outcome.label}
            description={workflow.outcome.description}
            icon={workflow.outcome.icon}
            isLast
            extra={
              <div className="bg-emerald-50 border border-emerald-200 rounded-md px-3 py-2 print:bg-transparent print:border-slate-200">
                <p className="text-xs font-semibold text-emerald-700 print:text-[8pt] print:text-slate-900">
                  {workflow.outcome.metric}
                </p>
              </div>
            }
          />
        </div>
      </div>

      {/* Tags */}
      <div className="px-6 pb-5 flex flex-wrap gap-2 print-tags">
        {workflow.tags.map((tag) => (
          <span
            key={tag}
            className="text-xs font-medium bg-slate-100 text-slate-600 px-2.5 py-1 rounded-md print:bg-transparent"
          >
            {tag}
          </span>
        ))}
      </div>
    </motion.article>
  );
}
