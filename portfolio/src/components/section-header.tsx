"use client";

import { motion } from "framer-motion";

interface SectionHeaderProps {
  label: string;
  title: string;
  description?: string;
}

export function SectionHeader({
  label,
  title,
  description,
}: SectionHeaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.5 }}
      className="mb-12"
    >
      <p className="text-sm font-semibold uppercase tracking-widest text-accent mb-2 print:text-[9pt] print:mb-1">
        {label}
      </p>
      <h2 className="text-3xl font-bold tracking-tight text-slate-900 print-section-heading">
        {title}
      </h2>
      {description && (
        <p className="mt-3 max-w-2xl text-lg text-slate-500 leading-relaxed print:text-[10pt] print:mt-1">
          {description}
        </p>
      )}
    </motion.div>
  );
}
