"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ExternalLink, Briefcase, Code2, CheckCircle2, Clock, Archive } from "lucide-react";
import {
  professionalProjects,
  personalProjects,
  type Project,
  type ProfessionalProject,
  type PersonalProject,
} from "@/lib/portfolio-data";
import { SectionHeader } from "./section-header";

type Filter = "all" | "professional" | "personal";

const allProjects: Project[] = [
  ...professionalProjects,
  ...personalProjects,
];

const statusConfig = {
  live: { label: "Live", icon: CheckCircle2, color: "text-emerald-600 bg-emerald-50" },
  "in-progress": { label: "In Progress", icon: Clock, color: "text-amber-600 bg-amber-50" },
  archived: { label: "Archived", icon: Archive, color: "text-slate-500 bg-slate-100" },
};

function ProfessionalCard({ project }: { project: ProfessionalProject }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm hover:shadow-md transition-shadow h-full flex flex-col print-project-card print-avoid-break">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-accent-50 flex items-center justify-center print:bg-transparent">
            <Briefcase size={16} className="text-accent" />
          </div>
          <span className="text-xs font-semibold uppercase tracking-wider text-accent">
            Professional
          </span>
        </div>
        <span className="text-xs text-slate-400 font-mono print:text-[8pt]">
          {project.period}
        </span>
      </div>

      <h3 className="text-lg font-bold text-slate-900 mb-1 print:text-[11pt]">
        {project.title}
      </h3>
      <p className="text-xs text-slate-400 mb-3 print:text-[8pt]">
        {project.client} &middot; {project.role}
      </p>
      <p className="text-sm text-slate-600 leading-relaxed mb-4 flex-grow print:text-[9pt]">
        {project.description}
      </p>

      {/* Impact */}
      <div className="mb-4">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
          Impact
        </p>
        <ul className="space-y-1.5">
          {project.impact.map((item, i) => (
            <li
              key={i}
              className="text-sm text-slate-700 flex items-start gap-2 print:text-[8pt]"
            >
              <CheckCircle2
                size={14}
                className="text-emerald-500 mt-0.5 shrink-0"
              />
              {item}
            </li>
          ))}
        </ul>
      </div>

      {/* Technologies */}
      <div className="flex flex-wrap gap-1.5 mt-auto print-tags">
        {project.technologies.map((tech) => (
          <span
            key={tech}
            className="text-[11px] font-medium bg-slate-100 text-slate-600 px-2 py-0.5 rounded print:bg-transparent"
          >
            {tech}
          </span>
        ))}
      </div>
    </div>
  );
}

function PersonalCard({ project }: { project: PersonalProject }) {
  const status = statusConfig[project.status];
  const StatusIcon = status.icon;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm hover:shadow-md transition-shadow h-full flex flex-col print-project-card print-avoid-break">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center print:bg-transparent">
            <Code2 size={16} className="text-slate-600" />
          </div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            Personal
          </span>
        </div>
        <span
          className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full ${status.color} print:bg-transparent`}
        >
          <StatusIcon size={12} />
          {status.label}
        </span>
      </div>

      <h3 className="text-lg font-bold text-slate-900 mb-2 print:text-[11pt]">
        {project.title}
      </h3>
      <p className="text-sm text-slate-600 leading-relaxed mb-4 flex-grow print:text-[9pt]">
        {project.description}
      </p>

      {/* Technologies */}
      <div className="flex flex-wrap gap-1.5 mt-auto print-tags">
        {project.technologies.map((tech) => (
          <span
            key={tech}
            className="text-[11px] font-medium bg-slate-100 text-slate-600 px-2 py-0.5 rounded print:bg-transparent"
          >
            {tech}
          </span>
        ))}
      </div>

      {project.link && (
        <a
          href={project.link}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-3 inline-flex items-center gap-1 text-sm text-accent hover:underline no-print"
        >
          View Project <ExternalLink size={14} />
        </a>
      )}
    </div>
  );
}

export function ProjectGrid() {
  const [filter, setFilter] = useState<Filter>("all");

  const filtered =
    filter === "all"
      ? allProjects
      : allProjects.filter((p) => p.category === filter);

  return (
    <section id="projects" className="py-24 px-6 bg-slate-50 print-bg-white print-compact print-page-break">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          label="Project Portfolio"
          title="Software Engineering & AI Projects"
          description="A curated selection of professional engagements and personal builds."
        />

        {/* Filter Tabs — hidden on print */}
        <div className="flex gap-2 mb-8 no-print">
          {(["all", "professional", "personal"] as Filter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors capitalize ${
                filter === f
                  ? "bg-slate-900 text-white"
                  : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-100"
              }`}
            >
              {f === "all" ? "All Projects" : f}
            </button>
          ))}
        </div>

        {/* Grid */}
        <AnimatePresence mode="wait">
          <motion.div
            key={filter}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6 print-single-col"
          >
            {filtered.map((project) =>
              project.category === "professional" ? (
                <ProfessionalCard
                  key={project.id}
                  project={project as ProfessionalProject}
                />
              ) : (
                <PersonalCard
                  key={project.id}
                  project={project as PersonalProject}
                />
              )
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
}
