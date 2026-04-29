"""Four LangChain tools the ReAct agent can call.

These are deliberately simple wrappers that surface deterministic,
explainable signals the model can use mid-thought. Together they
replace what would otherwise be RAG context — the agent reasons over
tool outputs, not over retrieved documents.
"""
import re

from langchain_core.tools import tool

from src.features import extract_all


@tool
def extract_red_flag_features(job_text: str) -> dict:
    """Run the heuristic feature extractor on a posting.

    Returns a dict of {signal_name: float} for the 7 cheap signals
    (urgency keywords, suspicious salary, all-caps ratio, etc.).
    """
    return extract_all(job_text)


# A small allowlist of well-known employers. Real production would use
# Crunchbase or SEC EDGAR; for this project a hardcoded list keeps the
# tool deterministic and offline.
KNOWN_COMPANIES = {
    "google", "microsoft", "apple", "amazon", "meta", "facebook",
    "netflix", "tesla", "stripe", "nvidia", "ibm", "intel", "oracle",
    "salesforce", "adobe", "sap", "vmware", "cisco", "uber", "airbnb",
    "spotify", "linkedin", "twitter", "x corp", "openai", "anthropic",
    "deloitte", "accenture", "kpmg", "ey", "pwc", "tcs", "infosys",
    "wipro",
}


@tool
def check_company_legitimacy(name: str) -> str:
    """Look up a company name against a list of well-known employers.

    Returns 'verified' or 'no record found'. The list is intentionally
    small — absence is informative, not damning.
    """
    if not isinstance(name, str):
        return "no record found"
    norm = name.lower().strip()
    for known in KNOWN_COMPANIES:
        if known in norm:
            return "verified"
    return "no record found"


# Annual salary benchmarks (USD) for common role families. Used to
# flag salaries wildly above market — those are usually scams. Below-
# market is NOT flagged here because legitimate startups underpay and
# legitimate offshore roles look low.
ROLE_BENCHMARKS = {
    "software": (60_000, 250_000),
    "engineer": (50_000, 220_000),
    "data": (60_000, 200_000),
    "designer": (40_000, 150_000),
    "marketing": (40_000, 140_000),
    "sales": (35_000, 200_000),
    "manager": (60_000, 250_000),
    "intern": (15_000, 70_000),
    "accountant": (40_000, 130_000),
    "writer": (25_000, 110_000),
    "support": (25_000, 80_000),
    "admin": (25_000, 80_000),
}


@tool
def analyze_salary_realism(text: str) -> str:
    """Compare a posted salary to a benchmark band for the role family.

    Pass a single string containing the role title, salary, and
    optionally location — any free-form format works
    (e.g. "Senior Backend Engineer, $180,000-$260,000, San Francisco").
    Returns 'plausible', 'suspiciously high', 'unparseable', or
    'no benchmark for role'.
    """
    text = text or ""
    lower = text.lower()
    # require at least one digit to avoid matching lone commas
    nums = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text)]
    if not nums:
        return "unparseable"
    posted = max(nums)
    for key, (_, hi) in ROLE_BENCHMARKS.items():
        if key in lower:
            return "suspiciously high" if posted > hi * 2 else "plausible"
    return "no benchmark for role"


FREE_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "proton.me", "protonmail.com", "live.com", "aol.com",
    "icloud.com", "rediffmail.com",
}


@tool
def check_email_domain(email: str) -> str:
    """Classify a contact email as 'free webmail' or 'corporate'.

    Real companies hire from corporate domains; gmail-based recruiting
    is a near-perfect scam tell.
    """
    if not isinstance(email, str) or "@" not in email:
        return "no email provided"
    domain = email.rsplit("@", 1)[1].lower().strip()
    return "free webmail" if domain in FREE_DOMAINS else "corporate"


# Convenience export for the agent at STEP 12.
ALL_TOOLS = [
    extract_red_flag_features,
    check_company_legitimacy,
    analyze_salary_realism,
    check_email_domain,
]
