"use client";

import { motion } from "framer-motion";
import { dashboardMetrics } from "@/lib/portfolio-data";
import { IconResolver } from "./icon-resolver";
import { SectionHeader } from "./section-header";

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

export function Dashboard() {
  return (
    <section id="dashboard" className="py-24 px-6 bg-slate-50 print-bg-white print-compact">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          label="Performance Overview"
          title="Last 12 Months at a Glance"
          description="Key delivery metrics across AI engineering, automation, and client engagement."
        />

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-80px" }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 print-metrics-row"
        >
          {dashboardMetrics.map((metric) => (
            <motion.div
              key={metric.label}
              variants={item}
              className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm hover:shadow-md transition-shadow print-avoid-break"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-lg bg-accent-50 flex items-center justify-center print:bg-transparent">
                  <IconResolver
                    name={metric.icon}
                    size={20}
                    className="text-accent"
                  />
                </div>
                <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-1 rounded-full print:bg-transparent print:text-[8pt]">
                  {metric.delta}
                </span>
              </div>
              <p className="text-3xl font-bold text-slate-900 tracking-tight print:text-[14pt]">
                {metric.value}
              </p>
              <p className="text-sm text-slate-500 mt-1 print:text-[9pt]">
                {metric.label}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
