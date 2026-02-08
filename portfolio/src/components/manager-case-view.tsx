"use client";

import { motion } from "framer-motion";
import { managerCaseStudy } from "@/lib/portfolio-data";
import { IconResolver } from "./icon-resolver";
import { SectionHeader } from "./section-header";

export function ManagerCaseView() {
  return (
    <section id="case-view" className="py-24 px-6 print-compact print-page-break">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          label="Leadership & Delivery"
          title={managerCaseStudy.title}
          description={managerCaseStudy.summary}
        />

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-10 print-single-col">
          {/* Principles sidebar */}
          <motion.aside
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-80px" }}
            transition={{ duration: 0.5 }}
            className="lg:col-span-1"
          >
            <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-400 mb-4 print:text-[9pt]">
              Core Principles
            </h3>
            <div className="space-y-3">
              {managerCaseStudy.principles.map((principle) => (
                <div
                  key={principle.label}
                  className="flex items-center gap-3 p-3 rounded-lg bg-slate-50 border border-slate-100 print:bg-transparent print:border-slate-200 print:p-2 print-avoid-break"
                >
                  <div className="w-8 h-8 rounded-md bg-accent-50 flex items-center justify-center shrink-0 print:bg-transparent">
                    <IconResolver
                      name={principle.icon}
                      size={16}
                      className="text-accent"
                    />
                  </div>
                  <span className="text-sm font-medium text-slate-700 print:text-[9pt]">
                    {principle.label}
                  </span>
                </div>
              ))}
            </div>
          </motion.aside>

          {/* Case Study Content */}
          <div className="lg:col-span-3 space-y-8">
            {managerCaseStudy.sections.map((section, index) => (
              <motion.div
                key={section.heading}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ duration: 0.5, delay: index * 0.08 }}
                className="print-case-section print-avoid-break"
              >
                <h3 className="text-lg font-bold text-slate-900 mb-3 flex items-center gap-3 print:text-[11pt]">
                  <span className="w-1.5 h-6 bg-accent rounded-full shrink-0 print:bg-accent" />
                  {section.heading}
                </h3>
                <p className="text-slate-600 leading-relaxed text-[15px] print:text-[9.5pt] print:leading-[1.5]">
                  {section.content}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
