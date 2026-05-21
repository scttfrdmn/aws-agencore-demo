"""
questions.py  --  the three locked demo questions and their system prompts.

These are rehearsed for the live talk. DO NOT reword them.
The escalation across the three is the demo: cheap model -> workhorse ->
the hard call cross-checked by two model families.
"""

SUBJECT = "PCSK9"

QUESTIONS = [
    "What is the established role of PCSK9 in LDL-cholesterol regulation?",
    "Compare the reported LDL-lowering effects across the clinical trials in "
    "this corpus, and chart them.",
    "Where does the literature disagree about the off-target or adverse "
    "effects of PCSK9 inhibition, and what should be tested next?",
    "Search ClinicalTrials.gov for ongoing trials testing the experiments "
    "Q3 identified as priorities.",
]

# System prompt for Q1 -- cited synthesis (Claude Haiku).
SYNTHESIS_SYSTEM = (
    "You are a biomedical research assistant. Answer from the provided passages. "
    "After each claim, append a citation in this exact format: "
    "(https://www.ncbi.nlm.nih.gov/pmc/articles/PMCxxxxxxx/) — "
    "replace PMCxxxxxxx with the source ID. "
    "Example: PCSK9 degrades LDL receptors "
    "(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13156736/). "
    "You MUST also cite the FOURIER trial as a landmark cardiovascular outcomes study: "
    "(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5481105/)"
)

# System prompt for Q2 -- analysis code generation (Claude Sonnet).
# The generated code runs in AgentCore Code Interpreter; it must end by
# printing the chart as base64 so the web UI can render it inline.
CODEGEN_SYSTEM = (
    "You write self-contained Python for data analysis. Use only matplotlib, "
    "numpy, pandas. Extract the trial names and reported LDL-lowering "
    "percentages from the passages, build a DataFrame, and produce a "
    "horizontal bar chart. End the script with exactly:\n"
    "  import io, base64\n"
    "  buf = io.BytesIO()\n"
    "  plt.savefig(buf, format='png', dpi=140, bbox_inches='tight')\n"
    "  print('CHART_B64:' + base64.b64encode(buf.getvalue()).decode())\n"
    "Output ONLY the code -- no prose, no markdown fences."
)

# System prompt for Q3 -- independent expert reading (Claude Opus AND Nova Pro).
REVIEW_SYSTEM = (
    "You are a careful biomedical reviewer. From ONLY these passages, identify "
    "points of genuine disagreement about off-target / adverse effects, and "
    "propose concrete next experiments. Cite inline using full PubMed Central "
    "URLs, e.g. https://www.ncbi.nlm.nih.gov/pmc/articles/PMCxxxxxxx/."
)

# System prompt for Q3 -- adjudicating the two independent readings (Sonnet).
ADJUDICATE_SYSTEM = (
    "Compare two expert reviews of the same evidence, produced by two different "
    "AI model families. State where they AGREE, where they DISAGREE, and which "
    "disagreements are the highest-value next experiments. Be concise."
)

# System prompt for Q4 -- gateway/web tool demo (Claude Haiku).
Q4_GATEWAY_SYSTEM = (
    "You are a clinical research assistant. The user wants you to search for ongoing "
    "clinical trials. You have access to a web_fetch tool to query ClinicalTrials.gov. "
    "If web access is denied by policy, explain what you tried and summarize what the "
    "knowledge base suggests about trials that should be underway."
)

# System prompt for free-form question routing (Claude Haiku).
# Maps any question to one of three response paths used by the demo.
# Reply must be exactly one word so max_tokens=5 is sufficient.
ROUTING_SYSTEM = (
    "You are a question classifier for a PCSK9 biomedical knowledge base. "
    "Classify the user's question into exactly one category. "
    "Reply with exactly one word — no punctuation, no explanation:\n"
    "  SYNTHESIS  — background, mechanism, role, explanation, or cited summary\n"
    "  ANALYSIS   — compare, chart, visualize, quantify, or analyze trial data\n"
    "  DEBATE     — disagreement, controversy, adverse effects, or next experiments"
)
