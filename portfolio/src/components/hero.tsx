"use client";

import { motion } from "framer-motion";
import { ArrowDown, Sparkles } from "lucide-react";

export function Hero() {
  return (
    <section className="relative min-h-[85vh] flex items-center justify-center hero-gradient overflow-hidden print-hero print-bg-white print-compact">
      {/* Decorative dot grid — hidden on print */}
      <div className="absolute inset-0 dot-pattern opacity-[0.04] no-print" />

      {/* Accent glow — hidden on print */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-accent/10 blur-[120px] no-print" />

      <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-slate-700 bg-slate-800/60 text-slate-300 text-sm mb-8 no-print"
        >
          <Sparkles size={14} className="text-accent" />
          AI Engineering &middot; Software Development &middot; Strategic
          Delivery
        </motion.div>

        {/* Heading */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight text-white leading-[1.1] print:text-slate-900"
        >
          Engineering{" "}
          <span className="text-accent accent-underline">Business Value</span>
          <br />
          through AI
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.25 }}
          className="mt-6 text-lg md:text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed print:text-slate-600"
        >
          Full-stack engineer and AI practitioner delivering production-grade
          intelligent systems. Specialising in LLM orchestration, document AI,
          and data-driven decision platforms for enterprise clients.
        </motion.p>

        {/* CTA — hidden on print */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.45 }}
          className="mt-10 flex items-center justify-center gap-4 no-print"
        >
          <a
            href="#dashboard"
            className="inline-flex items-center gap-2 px-6 py-3 bg-accent hover:bg-accent-700 text-white font-medium rounded-lg transition-colors"
          >
            View My Work
            <ArrowDown size={16} />
          </a>
          <button
            onClick={() => window.print()}
            className="inline-flex items-center gap-2 px-6 py-3 border border-slate-600 text-slate-300 hover:bg-slate-800 font-medium rounded-lg transition-colors"
          >
            Download PDF
          </button>
        </motion.div>
      </div>

      {/* Print-only header info */}
      <div className="hidden print-only print:block print:absolute print:bottom-0 print:left-0 print:right-0 print:text-center">
        <p className="text-[9pt] text-slate-400 border-t border-slate-200 pt-2 mt-4">
          Executive Portfolio &middot; Confidential
        </p>
      </div>
    </section>
  );
}
