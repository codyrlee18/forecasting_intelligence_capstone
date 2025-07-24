"""
Synthetic SaaS datasets generator
--------------------------------
Generates four CSV files in the current working directory:
    1. product_usage.csv
    2. crm.csv
    3. finance.csv
    4. survey.csv
Creates ≈68 k rows of time‑series data (24 months) for ~2 800 accounts.
Dependencies: pandas, numpy, faker
Run:  pip install pandas numpy faker && python generate_saas_datasets.py
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import timedelta
from pathlib import Path

# ----------------------------- configuration ---------------------------------
SEED = 42
rng = np.random.default_rng(SEED)
fake = Faker()
Faker.seed(SEED)

N_ACCOUNTS = 2800  # ≈67 k rows over 24 months
MONTHS = pd.period_range("2023-01", periods=24, freq="M")

INDUSTRIES = [
    "FinTech", "HealthTech", "EdTech", "MarTech", "E‑Commerce",
    "CyberSecurity", "Logistics", "SMB SaaS",
]
FEATURES = ["Core", "Analytics", "API", "Automation", "Premium Support"]
DEAL_STATUS_VALUES = ["Closed‑Won", "Negotiation", "Closed‑Lost", "On‑Hold"]
CUSTOMER_TYPES = ["New", "Expansion"]

# ----------------------------- helper utils ----------------------------------

def make_account_id(idx: int) -> str:
    return f"A{idx:05d}"


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

# ------------------------------ core tables ----------------------------------
accounts = pd.DataFrame({
    "account_id": [make_account_id(i) for i in range(1, N_ACCOUNTS + 1)],
    "account_name": [fake.company() for _ in range(N_ACCOUNTS)],
    "industry": rng.choice(INDUSTRIES, size=N_ACCOUNTS),
    "customer_type": rng.choice(CUSTOMER_TYPES, size=N_ACCOUNTS, p=[0.7, 0.3]),
    "contract_length_months": rng.integers(6, 36, size=N_ACCOUNTS),
})

accounts["client_tenure_months"] = rng.integers(1, 60, size=N_ACCOUNTS)

# ------------------------- monthly product usage -----------------------------
usage_rows = []
for period in MONTHS:
    base_date = period.end_time
    days_in_month = base_date.day

    usage_rows.append(pd.DataFrame({
        "account_id": accounts.account_id.values,
        "date": base_date.strftime("%Y-%m-%d"),
        "feature_purchased": rng.choice(FEATURES, size=N_ACCOUNTS),
        "sessions_last_30d": rng.poisson(lam=8, size=N_ACCOUNTS),
        "last_login_days": rng.integers(0, days_in_month, size=N_ACCOUNTS),
        "support_tickets_30d": rng.poisson(lam=0.3, size=N_ACCOUNTS),
    }))

product_usage = pd.concat(usage_rows, ignore_index=True)

# ------------------------------ survey data ----------------------------------
csat_base = rng.uniform(6.5, 8.5, size=N_ACCOUNTS)
survey_rows = []
for idx, acc in accounts.iterrows():
    for period in MONTHS:
        survey_rows.append({
            "account_id": acc.account_id,
            "date": (period.end_time - timedelta(days=int(rng.integers(0, 10)))).strftime("%Y-%m-%d"),
            "csat_score": round(np.clip(rng.normal(csat_base[idx], 1.0), 1, 10), 1),
            "comment_sentiment_score": np.round(rng.normal((csat_base[idx] - 5) / 5, 0.3), 2),
        })

aud_survey = pd.DataFrame(survey_rows)

# ------------------------ churn mechanism & finance --------------------------
finance_rows = []
for period in MONTHS:
    base_date = period.end_time

    baseline_rev = rng.gamma(shape=2.0, scale=1200, size=N_ACCOUNTS)

    mask_usage = product_usage.date == base_date.strftime("%Y-%m-%d")
    sessions = product_usage.loc[mask_usage, "sessions_last_30d"].values
    last_login = product_usage.loc[mask_usage, "last_login_days"].values

    csat_month = (
        aud_survey[aud_survey.date <= base_date.strftime("%Y-%m-%d")]
        .groupby("account_id").last()
        .reindex(accounts.account_id).csat_score.values
    )

    # increase churn rate by slightly raising base probability
    score = (
        -0.03 * sessions +
        0.05 * last_login -
        0.4 * (csat_month - 5) +
        rng.normal(0.5, 0.5, N_ACCOUNTS)  # slight upward bias
    )
    churn_flag = rng.random(N_ACCOUNTS) < sigmoid(score)

    closed_won = baseline_rev
    churned = closed_won * churn_flag
    actual = closed_won - churned

    finance_rows.append(pd.DataFrame({
        "account_id": accounts.account_id.values,
        "month": base_date.strftime("%Y-%m-%d"),
        "closed_won_usd": closed_won.round(2),
        "churned_usd": churned.round(2),
        "actual_usd": actual.round(2),
        "churn": churn_flag.astype(int),
        "active_customer": (~churn_flag).astype(int),
    }))

finance = pd.concat(finance_rows, ignore_index=True)

# ------------------------- revenue scaling 1‑3 M -----------------------------
for period in MONTHS:
    eom_str = period.end_time.strftime("%Y-%m-%d")
    mask = finance.month == eom_str
    total = finance.loc[mask, "actual_usd"].sum()
    factor = rng.uniform(1_000_000, 3_000_000) / (total if total else 1)
    finance.loc[mask, ["closed_won_usd", "churned_usd", "actual_usd"]] = (
        finance.loc[mask, ["closed_won_usd", "churned_usd", "actual_usd"]] * factor
    ).round(2)

# ------------------------------ crm dataset ----------------------------------
crm = accounts.copy()
crm["deal_value_usd"] = rng.normal(15_000, 4_000, N_ACCOUNTS).clip(2_000).round(2)
crm["deal_status"] = rng.choice(
    DEAL_STATUS_VALUES, size=N_ACCOUNTS, p=[0.8, 0.12, 0.05, 0.03]
)

# --------------------------------- output ------------------------------------
out_dir = Path.cwd()
product_usage.to_csv(out_dir / "product_usage.csv", index=False)
crm.to_csv(out_dir / "crm.csv", index=False)
finance.to_csv(out_dir / "finance.csv", index=False)
aud_survey.to_csv(out_dir / "survey.csv", index=False)

print("Datasets generated →")
for f in ("product_usage.csv", "crm.csv", "finance.csv", "survey.csv"):
    print(f" • {f}")
