"""Heuristic features for TrustLens.

Each function takes the posting text (the tagged ``job_text`` built in
``notebooks/01_eda.ipynb``) and returns a single float. ``extract_all``
runs every function and returns a name->float dict so the rest of the
project only has to call one thing.

These features are deliberately cheap and explainable. They feed two
things downstream:
- the LLM prompt at STEP 8/12, as a small numeric block ("signals my
  code already noticed"), and
- the non-LLM DistilBERT baseline at STEP 9, so the rubric can compare
  the fine-tuned model against rules.
"""
import re

URGENCY_WORDS = [
    "urgent", "immediate", "asap", "right away",
    "today only", "act now", "limited time", "hurry",
    "apply now", "start tomorrow", "no time to waste",
]

PAYMENT_PHRASES = [
    "registration fee", "training fee", "deposit",
    "pay upfront", "send money", "wire transfer",
    "processing fee", "starter kit", "pay to start",
    "send funds", "pay for materials",
]

FREE_EMAIL_DOMAINS = (
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "proton.me", "protonmail.com", "live.com", "aol.com",
    "rediffmail.com", "icloud.com",
)

MESSENGER_HINTS = ("whatsapp", "telegram", "wechat", "signal app", "skype me")

# matches "$5000/week", "$500 per day", "$150/hour" etc.
HIGH_PAY_RE = re.compile(
    r"\$\s?(\d{2,5})(?:[.,]\d{1,3})?\s*(?:per\s*|/)?\s*(week|day|hour|hr)\b",
    re.IGNORECASE,
)

EMAIL_RE = re.compile(r"[\w.+-]+@([\w.-]+\.\w+)", re.IGNORECASE)


def salary_range_pattern(text: str) -> float:
    """1.0 if the posting promises a suspiciously high pay rate, else 0.0.

    Real engineering salaries are quoted annually, so they don't show up
    in this regex at all. We only flag /week, /day, /hour rates above
    thresholds that real legit work-from-home jobs almost never pay.
    """
    for match in HIGH_PAY_RE.finditer(text):
        amount = int(match.group(1))
        unit = match.group(2).lower()
        if unit == "week" and amount >= 3000:
            return 1.0
        if unit == "day" and amount >= 500:
            return 1.0
        if unit in ("hour", "hr") and amount >= 150:
            return 1.0
    return 0.0


def urgency_keywords(text: str) -> float:
    """Count of urgency phrases. Scams pressure you to act before you think."""
    lower = text.lower()
    return float(sum(1 for w in URGENCY_WORDS if w in lower))


def suspicious_contact(text: str) -> float:
    """1.0 if contact routes through free webmail or a messenger app.

    Real companies hire from corporate email domains. Recruiters who
    move you to WhatsApp/Telegram are a near-perfect scam tell.
    """
    lower = text.lower()
    for hint in MESSENGER_HINTS:
        if hint in lower:
            return 1.0
    for match in EMAIL_RE.finditer(text):
        if match.group(1).lower() in FREE_EMAIL_DOMAINS:
            return 1.0
    return 0.0


def missing_company_signals(text: str) -> float:
    """1.0 if the [COMPANY PROFILE] section is missing or near-empty.

    The EDA notebook tags fields with [COMPANY PROFILE]. A missing tag
    means the source CSV had no company description; a tag with <30
    chars of content is essentially the same situation.
    """
    if "[COMPANY PROFILE]" not in text:
        return 1.0
    after = text.split("[COMPANY PROFILE]", 1)[1]
    next_tag = re.search(r"\n\[[A-Z ]+\]", after)
    section = after[: next_tag.start()] if next_tag else after
    return 1.0 if len(section.strip()) < 30 else 0.0


def payment_upfront_keywords(text: str) -> float:
    """Count of phrases that imply the candidate has to pay to get the job."""
    lower = text.lower()
    return float(sum(1 for p in PAYMENT_PHRASES if p in lower))


def allcaps_ratio(text: str) -> float:
    """Fraction of >=3-letter words that are ALL CAPS.

    The 3-letter floor excludes acronyms like IT, HR, US that would
    otherwise inflate the ratio for normal postings.
    """
    words = re.findall(r"\b[A-Za-z]{3,}\b", text)
    if not words:
        return 0.0
    caps = sum(1 for w in words if w.isupper())
    return caps / len(words)


def exclamation_density(text: str) -> float:
    """Exclamation marks per 1000 characters. Length-normalised so a
    long real posting with one ``!`` doesn't look the same as a short
    ad with five.
    """
    if not text:
        return 0.0
    return text.count("!") * 1000.0 / len(text)


def extract_all(text: str) -> dict:
    """Run every feature on the posting text. Returns name->float dict."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return {
        "salary_range_pattern": salary_range_pattern(text),
        "urgency_keywords": urgency_keywords(text),
        "suspicious_contact": suspicious_contact(text),
        "missing_company_signals": missing_company_signals(text),
        "payment_upfront_keywords": payment_upfront_keywords(text),
        "allcaps_ratio": allcaps_ratio(text),
        "exclamation_density": exclamation_density(text),
    }
