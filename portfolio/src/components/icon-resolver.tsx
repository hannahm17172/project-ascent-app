"use client";

import {
  AlertTriangle,
  BarChart3,
  BookOpen,
  Bot,
  Brain,
  Briefcase,
  Clock,
  Cog,
  Cpu,
  Layers,
  RefreshCw,
  Target,
  TrendingUp,
  Users,
  type LucideIcon,
} from "lucide-react";

const iconMap: Record<string, LucideIcon> = {
  "alert-triangle": AlertTriangle,
  "bar-chart-3": BarChart3,
  "book-open": BookOpen,
  bot: Bot,
  brain: Brain,
  briefcase: Briefcase,
  clock: Clock,
  cog: Cog,
  cpu: Cpu,
  layers: Layers,
  "refresh-cw": RefreshCw,
  target: Target,
  "trending-up": TrendingUp,
  users: Users,
};

interface IconResolverProps {
  name: string;
  className?: string;
  size?: number;
}

export function IconResolver({
  name,
  className = "",
  size = 20,
}: IconResolverProps) {
  const Icon = iconMap[name];
  if (!Icon) return null;
  return <Icon className={className} size={size} />;
}
