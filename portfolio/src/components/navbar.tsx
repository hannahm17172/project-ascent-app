"use client";

import { useState, useEffect } from "react";
import { Zap } from "lucide-react";

const navLinks = [
  { label: "Dashboard", href: "#dashboard" },
  { label: "AI Workflows", href: "#workflows" },
  { label: "Projects", href: "#projects" },
  { label: "Case View", href: "#case-view" },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav
      className={`no-print fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? "bg-white/90 backdrop-blur-md border-b border-slate-200 shadow-sm"
          : "bg-transparent"
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <a href="#" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
            <Zap size={16} className="text-white" />
          </div>
          <span
            className={`font-bold text-sm tracking-tight transition-colors ${
              scrolled ? "text-slate-900" : "text-white"
            }`}
          >
            Portfolio
          </span>
        </a>

        {/* Links */}
        <div className="hidden md:flex items-center gap-6">
          {navLinks.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className={`text-sm font-medium transition-colors hover:text-accent ${
                scrolled ? "text-slate-600" : "text-slate-300"
              }`}
            >
              {link.label}
            </a>
          ))}
        </div>

        {/* Print CTA */}
        <button
          onClick={() => window.print()}
          className={`text-sm font-medium px-4 py-2 rounded-lg transition-colors ${
            scrolled
              ? "bg-slate-900 text-white hover:bg-slate-800"
              : "bg-white/10 text-white border border-white/20 hover:bg-white/20"
          }`}
        >
          Download PDF
        </button>
      </div>
    </nav>
  );
}
