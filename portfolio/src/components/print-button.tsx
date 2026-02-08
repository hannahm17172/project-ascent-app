"use client";

import { Printer } from "lucide-react";

export function PrintButton() {
  return (
    <button
      onClick={() => window.print()}
      className="no-print fixed bottom-6 right-6 z-50 inline-flex items-center gap-2 px-4 py-3 bg-slate-900 hover:bg-slate-800 text-white text-sm font-medium rounded-lg shadow-lg transition-colors"
      aria-label="Print or save as PDF"
    >
      <Printer size={16} />
      <span className="hidden sm:inline">Export PDF</span>
    </button>
  );
}
