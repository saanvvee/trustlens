
from src.features import (
    salary_range_pattern,
    urgency_keywords,
    suspicious_contact,
    missing_company_signals,
    payment_upfront_keywords,
    allcaps_ratio,
    exclamation_density,
    extract_all,
)


def test_salary_high_weekly_fires():
    assert salary_range_pattern("Earn $5000/week from home!") == 1.0


def test_salary_normal_annual_silent():
    # annual salaries do not match the /week, /day, /hour regex at all
    assert salary_range_pattern("Salary range $90,000 - $120,000 per year") == 0.0


def test_urgency_counts_multiple_phrases():
    # "urgent" + "immediate" + "apply now" -> 3
    assert urgency_keywords("URGENT immediate hire, apply now!") == 3.0


def test_urgency_silent_on_neutral_text():
    assert urgency_keywords("We are looking for a software engineer.") == 0.0


def test_contact_flags_free_email():
    assert suspicious_contact("Email recruiter@gmail.com to apply.") == 1.0


def test_contact_flags_whatsapp():
    assert suspicious_contact("Reply on WhatsApp at +1234567890") == 1.0


def test_contact_silent_on_corporate_email():
    assert suspicious_contact("Email careers@stripe.com") == 0.0


def test_missing_company_fires_when_tag_absent():
    assert missing_company_signals("[TITLE] Customer rep\n[DESCRIPTION] do stuff") == 1.0


def test_missing_company_silent_when_section_present():
    text = (
        "[TITLE] X\n"
        "[COMPANY PROFILE] Founded in 2010, we serve millions of customers worldwide.\n"
        "[DESCRIPTION] role details"
    )
    assert missing_company_signals(text) == 0.0


def test_payment_counts_three_phrases():
    # "registration fee" + "starter kit" + "deposit"
    assert payment_upfront_keywords("Pay a registration fee and starter kit deposit") == 3.0


def test_allcaps_high_on_shouty():
    # HIRE, FAST, NOW are caps; "cash" is not -> 3/4
    val = allcaps_ratio("HIRE FAST cash NOW")
    assert 0.7 < val <= 1.0


def test_allcaps_zero_on_normal_text():
    assert allcaps_ratio("This is a normal job posting with no shouting.") == 0.0


def test_exclamation_density_zero_when_none():
    assert exclamation_density("hello world") == 0.0


def test_exclamation_density_positive_when_present():
    assert exclamation_density("hello!!!") > 0


def test_extract_all_returns_seven_floats():
    out = extract_all("Apply now! Send $5000/week to recruiter@gmail.com")
    expected_keys = {
        "salary_range_pattern", "urgency_keywords", "suspicious_contact",
        "missing_company_signals", "payment_upfront_keywords",
        "allcaps_ratio", "exclamation_density",
    }
    assert set(out.keys()) == expected_keys
    for v in out.values():
        assert isinstance(v, float)


def test_extract_all_handles_none_input():
    out = extract_all(None)
    # missing tag -> 1.0; everything else should be 0.0
    assert out["missing_company_signals"] == 1.0
    assert out["urgency_keywords"] == 0.0
    assert out["allcaps_ratio"] == 0.0
