// ============================================================================
// PORTFOLIO DATA — Single source of truth for the entire application.
// Edit this file to update all sections of the portfolio.
// ============================================================================

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DashboardMetric {
  label: string;
  value: string;
  delta: string;
  icon: string;
}

export interface WorkflowStep {
  label: string;
  description: string;
  icon: string;
}

export interface AIWorkflow {
  id: string;
  title: string;
  subtitle: string;
  problem: WorkflowStep;
  agent: WorkflowStep & { model: string };
  process: WorkflowStep & { steps: string[] };
  outcome: WorkflowStep & { metric: string };
  tags: string[];
}

export interface ProfessionalProject {
  id: string;
  title: string;
  client: string;
  role: string;
  period: string;
  description: string;
  impact: string[];
  technologies: string[];
  category: "professional";
}

export interface PersonalProject {
  id: string;
  title: string;
  description: string;
  technologies: string[];
  link?: string;
  status: "live" | "in-progress" | "archived";
  category: "personal";
}

export type Project = ProfessionalProject | PersonalProject;

export interface CaseSection {
  heading: string;
  content: string;
}

export interface ManagerCaseStudy {
  title: string;
  summary: string;
  sections: CaseSection[];
  principles: { label: string; icon: string }[];
}

// ---------------------------------------------------------------------------
// Dashboard Metrics — Last 12 Months
// ---------------------------------------------------------------------------

export const dashboardMetrics: DashboardMetric[] = [
  {
    label: "Projects Delivered",
    value: "12",
    delta: "+3 YoY",
    icon: "briefcase",
  },
  {
    label: "AI Pipelines Deployed",
    value: "4",
    delta: "Production-Grade",
    icon: "cpu",
  },
  {
    label: "Hours Automated",
    value: "2,400+",
    delta: "Per Quarter",
    icon: "clock",
  },
  {
    label: "Stakeholder Satisfaction",
    value: "98%",
    delta: "Avg. CSAT",
    icon: "trending-up",
  },
];

// ---------------------------------------------------------------------------
// AI Workflows — The "Meat" of the Portfolio
// ---------------------------------------------------------------------------

export const aiWorkflows: AIWorkflow[] = [
  {
    id: "wf-1",
    title: "Enterprise Document Intelligence Pipeline",
    subtitle:
      "Transforming manual document processing into an automated, AI-driven pipeline for a Fortune 500 financial services client.",
    problem: {
      label: "Manual Extraction",
      description:
        "Analysts spent 6+ hours daily extracting data from unstructured financial documents — invoices, contracts, and compliance forms — with a 12% error rate.",
      icon: "alert-triangle",
    },
    agent: {
      label: "Multi-Model Orchestrator",
      description:
        "Deployed a LangChain agent that routes documents to specialised models based on document type and complexity.",
      model: "GPT-4 Turbo + Azure Document Intelligence",
      icon: "bot",
    },
    process: {
      label: "Intelligent Pipeline",
      description:
        "End-to-end automated workflow from ingestion to structured output.",
      steps: [
        "Document ingestion & classification (40+ types)",
        "Entity extraction via fine-tuned NER models",
        "Cross-validation against compliance rules",
        "Structured JSON output to downstream systems",
      ],
      icon: "cog",
    },
    outcome: {
      label: "Measurable ROI",
      description:
        "Delivered production system processing 3,000+ documents daily.",
      metric: "78% reduction in processing time · $1.2M annual savings",
      icon: "trending-up",
    },
    tags: ["NLP", "Document AI", "Enterprise", "LangChain", "Azure"],
  },
  {
    id: "wf-2",
    title: "AI-Powered Risk Assessment Engine",
    subtitle:
      "Built a real-time risk scoring system that analyses regulatory filings and flags anomalies before human reviewers see them.",
    problem: {
      label: "Compliance Bottleneck",
      description:
        "Risk teams reviewed 500+ filings weekly with inconsistent scoring. Average turnaround was 4 business days, creating regulatory exposure.",
      icon: "alert-triangle",
    },
    agent: {
      label: "RAG-Enhanced Analyst",
      description:
        "Retrieval-Augmented Generation agent with access to internal policy documents and regulatory databases.",
      model: "GPT-4 + ChromaDB + Custom Embeddings",
      icon: "brain",
    },
    process: {
      label: "Automated Risk Scoring",
      description:
        "Multi-stage pipeline combining retrieval, analysis, and human-in-the-loop validation.",
      steps: [
        "Ingest filings and chunk for semantic search",
        "Retrieve relevant policy context via vector similarity",
        "Generate risk score with explainable reasoning",
        "Route high-risk items to senior reviewers",
      ],
      icon: "cog",
    },
    outcome: {
      label: "Speed & Accuracy",
      description:
        "Reduced review cycle from 4 days to same-day for 85% of filings.",
      metric: "85% faster turnaround · 94% accuracy on risk flags",
      icon: "trending-up",
    },
    tags: ["RAG", "Risk", "Compliance", "Vector DB", "Embeddings"],
  },
  {
    id: "wf-3",
    title: "Conversational Data Analytics Assistant",
    subtitle:
      "Designed a natural-language interface that lets non-technical stakeholders query complex datasets without writing SQL.",
    problem: {
      label: "Data Accessibility Gap",
      description:
        "Business stakeholders waited 2–3 days for ad-hoc data requests because querying required SQL expertise and analyst availability.",
      icon: "alert-triangle",
    },
    agent: {
      label: "Text-to-SQL Agent",
      description:
        "A conversational agent that translates natural language questions into optimised SQL queries with guardrails.",
      model: "GPT-4 + LangChain SQL Toolkit",
      icon: "bot",
    },
    process: {
      label: "Query & Visualise",
      description:
        "Intent parsing, query generation, execution, and chart rendering in a single conversation turn.",
      steps: [
        "Parse user intent and identify target tables",
        "Generate parameterised SQL with injection prevention",
        "Execute against read-replica with row-level security",
        "Render results as charts or summary tables",
      ],
      icon: "cog",
    },
    outcome: {
      label: "Self-Service Analytics",
      description:
        "Enabled 40+ non-technical users to independently access data insights.",
      metric: "90% reduction in ad-hoc analyst requests · 2-day → 2-minute answers",
      icon: "trending-up",
    },
    tags: ["Text-to-SQL", "Analytics", "Conversational AI", "Self-Service"],
  },
  {
    id: "wf-4",
    title: "Project Ascent: Custom AI Research Agent",
    subtitle:
      "Personal build — a Streamlit-based research agent that combines document RAG with live web search for deep-dive investigations.",
    problem: {
      label: "Fragmented Research",
      description:
        "Research across multiple PDFs, web sources, and internal docs required switching tools constantly, losing context with each transition.",
      icon: "alert-triangle",
    },
    agent: {
      label: "Hybrid Search Agent",
      description:
        "Custom agent combining vector-based document retrieval with real-time DuckDuckGo web search.",
      model: "Google Gemini + Sentence Transformers + ChromaDB",
      icon: "brain",
    },
    process: {
      label: "Unified Research Pipeline",
      description:
        "Upload documents, build a knowledge base, then ask questions that span both local docs and the open web.",
      steps: [
        "PDF/TXT ingestion with smart chunking",
        "Embedding generation via Sentence Transformers",
        "Hybrid retrieval: vector similarity + web search",
        "Context-aware synthesis with configurable LLM",
      ],
      icon: "cog",
    },
    outcome: {
      label: "Research Accelerator",
      description:
        "Open-source tool used for client research prep and personal learning.",
      metric: "60% faster research cycles · Unified context across sources",
      icon: "trending-up",
    },
    tags: ["RAG", "Streamlit", "Open Source", "Gemini", "Personal Build"],
  },
];

// ---------------------------------------------------------------------------
// Professional Projects
// ---------------------------------------------------------------------------

export const professionalProjects: ProfessionalProject[] = [
  {
    id: "pp-1",
    title: "Enterprise Document Intelligence Platform",
    client: "Fortune 500 Financial Services",
    role: "Technical Lead",
    period: "Q1 2024 – Q3 2024",
    description:
      "Designed and deployed an AI-powered document processing pipeline that extracts, classifies, and routes critical financial documents across the organisation. Led a team of 4 engineers from prototype to production.",
    impact: [
      "Reduced manual processing time by 78%",
      "Achieved 96% extraction accuracy across 40+ document types",
      "Saved $1.2M annually in operational costs",
    ],
    technologies: [
      "Python",
      "LangChain",
      "GPT-4",
      "Azure Document Intelligence",
      "FastAPI",
      "PostgreSQL",
    ],
    category: "professional",
  },
  {
    id: "pp-2",
    title: "Real-Time Risk Assessment Engine",
    client: "Global Insurance Provider",
    role: "AI/ML Engineer",
    period: "Q2 2024 – Q4 2024",
    description:
      "Built a real-time risk scoring system that analyses regulatory filings and flags anomalies before human reviewers see them. Integrated RAG pipeline with internal policy databases.",
    impact: [
      "85% faster review turnaround (4 days → same-day)",
      "94% accuracy on automated risk flags",
      "Processed 500+ filings weekly with consistent scoring",
    ],
    technologies: [
      "Python",
      "GPT-4",
      "ChromaDB",
      "FastAPI",
      "React",
      "Docker",
    ],
    category: "professional",
  },
  {
    id: "pp-3",
    title: "Conversational Analytics Dashboard",
    client: "Retail & Consumer Goods",
    role: "Full-Stack Engineer",
    period: "Q3 2024 – Q1 2025",
    description:
      "Designed a natural-language interface enabling non-technical stakeholders to query complex retail datasets without SQL expertise. Built end-to-end from LLM agent to React frontend.",
    impact: [
      "90% reduction in ad-hoc analyst requests",
      "40+ non-technical users onboarded in first month",
      "Average query response time under 3 seconds",
    ],
    technologies: [
      "TypeScript",
      "Next.js",
      "LangChain",
      "GPT-4",
      "PostgreSQL",
      "Tailwind CSS",
    ],
    category: "professional",
  },
  {
    id: "pp-4",
    title: "Automated Compliance Reporting Pipeline",
    client: "Healthcare & Life Sciences",
    role: "Data Engineer",
    period: "Q4 2023 – Q2 2024",
    description:
      "Built an automated pipeline that ingests regulatory data, applies business rules, and generates audit-ready compliance reports — replacing a manual process that previously required a team of 6.",
    impact: [
      "Eliminated 120 hours/month of manual report generation",
      "Zero compliance misses since deployment",
      "Enabled real-time reporting for the first time",
    ],
    technologies: [
      "Python",
      "Apache Airflow",
      "dbt",
      "Snowflake",
      "Tableau",
      "AWS",
    ],
    category: "professional",
  },
];

// ---------------------------------------------------------------------------
// Personal Software Projects
// ---------------------------------------------------------------------------

export const personalProjects: PersonalProject[] = [
  {
    id: "ps-1",
    title: "Project Ascent — AI Research Agent",
    description:
      "A Streamlit-based research agent combining PDF document RAG with live web search. Supports configurable LLMs, custom system prompts, and persistent knowledge bases via ChromaDB.",
    technologies: [
      "Python",
      "Streamlit",
      "LangChain",
      "Google Gemini",
      "ChromaDB",
      "Sentence Transformers",
    ],
    status: "live",
    category: "personal",
  },
  {
    id: "ps-2",
    title: "Executive Portfolio — This Site",
    description:
      "A dual-mode digital portfolio built with Next.js 14 that transforms between an interactive web experience and a print-ready PDF case study document.",
    technologies: [
      "TypeScript",
      "Next.js 14",
      "Tailwind CSS",
      "Framer Motion",
      "Lucide React",
    ],
    status: "live",
    category: "personal",
  },
  {
    id: "ps-3",
    title: "CLI Task Orchestrator",
    description:
      "A terminal-based project management tool that lets developers define, track, and automate development workflows using YAML configuration files.",
    technologies: ["Go", "Cobra", "YAML", "SQLite"],
    status: "in-progress",
    category: "personal",
  },
  {
    id: "ps-4",
    title: "ML Model Performance Monitor",
    description:
      "A lightweight dashboard for tracking ML model drift, latency, and accuracy metrics in production environments. Designed for small teams without MLOps infrastructure.",
    technologies: ["Python", "FastAPI", "React", "D3.js", "Redis"],
    status: "in-progress",
    category: "personal",
  },
];

// ---------------------------------------------------------------------------
// Manager Case Study — Strategic Breakdown
// ---------------------------------------------------------------------------

export const managerCaseStudy: ManagerCaseStudy = {
  title: "How I Manage: Delivery Philosophy & Approach",
  summary:
    "I operate at the intersection of technical execution and stakeholder management. Every project I lead follows a structured methodology designed to maximise business value while minimising delivery risk.",
  sections: [
    {
      heading: "Agile Delivery with Business Guardrails",
      content:
        "I run two-week sprints anchored to business milestones, not just technical tasks. Every sprint planning session starts with 'What business question does this answer?' rather than 'What features do we build?' This keeps engineering effort tightly coupled to measurable outcomes. I maintain a rolling 6-week roadmap visible to all stakeholders, updated every sprint. This cadence gives leadership predictability while preserving the team's ability to adapt. When scope changes arise — and they always do — I use a structured impact assessment framework: effort, risk, dependency, and opportunity cost. This turns emotional debates into data-driven decisions.",
    },
    {
      heading: "Stakeholder Communication & Executive Alignment",
      content:
        "I believe the biggest risk to any technical project is not the technology — it's misaligned expectations. I run three communication tracks in parallel: (1) Weekly 15-minute executive briefings focused on outcomes and blockers, not technical details. (2) Bi-weekly deep-dives with technical stakeholders covering architecture decisions and trade-offs. (3) Async Slack/Teams updates for the broader team with clear status indicators. For executive audiences, I translate technical progress into business language. Instead of 'We deployed the embedding pipeline,' I report 'The system can now process 3,000 documents per day, up from 400 manually — on track for the Q3 cost savings target.'",
    },
    {
      heading: "Technical Decision-Making Framework",
      content:
        "I evaluate technology choices through four lenses: (1) Does it solve the immediate problem? (2) Can the team maintain it after I leave? (3) Does it integrate with the client's existing ecosystem? (4) What's the total cost of ownership over 18 months? This framework prevents over-engineering and 'resume-driven development.' I've actively chosen simpler solutions — like a well-structured Python script over a Kubernetes-orchestrated microservice — when the business context warranted it. The best architecture is the one the team can operate confidently.",
    },
    {
      heading: "Team Development & Knowledge Transfer",
      content:
        "Every project I lead includes a deliberate knowledge transfer plan from day one. I pair junior engineers with senior leads on critical path items, run weekly 'architecture walkthrough' sessions where team members present their own work, and maintain living documentation that evolves with the codebase. My goal is to make myself replaceable. If the project can't survive without me, I haven't done my job as a lead. I measure success not just by what we deliver, but by whether the client's team can extend and maintain it independently.",
    },
  ],
  principles: [
    { label: "Outcome-Driven Sprints", icon: "target" },
    { label: "Executive Communication", icon: "users" },
    { label: "Pragmatic Architecture", icon: "layers" },
    { label: "Knowledge Transfer", icon: "book-open" },
    { label: "Data-Driven Decisions", icon: "bar-chart-3" },
    { label: "Continuous Delivery", icon: "refresh-cw" },
  ],
};
