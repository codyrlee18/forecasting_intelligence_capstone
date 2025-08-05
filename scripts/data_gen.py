import numpy as np
import pandas as pd
from faker import Faker
from datetime import timedelta
from pathlib import Path
import csv

def data_generation():
    # ----------------------------- configuration ---------------------------------
    SEED = 42
    rng = np.random.default_rng(SEED)
    fake = Faker()
    Faker.seed(SEED)

    N_ACCOUNTS = 2800
    MONTHS = pd.period_range("2023-01", periods=24, freq="M")

    INDUSTRIES = [
        "FinTech", "HealthTech", "EdTech", "MarTech", "E -Commerce",
        "CyberSecurity", "Logistics", "SMB SaaS",
    ]
    FEATURES = ["Core", "Analytics", "API", "Automation", "Premium Support"]
    DEAL_STATUS_VALUES = ["Closed -Won", "Negotiation", "Closed -Lost", "On -Hold"]
    CUSTOMER_TYPES = ["New", "Expansion"]
    REGIONS = ["North America", "Europe", "LATAM", "Asia"]

    def make_account_id(idx: int) -> str:
        return f"A{idx:05d}"

    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    client_tenure = rng.integers(1, 60, size=N_ACCOUNTS)

    accounts = pd.DataFrame({
        "account_id": [make_account_id(i) for i in range(1, N_ACCOUNTS + 1)],
        "account_name": [fake.company() for _ in range(N_ACCOUNTS)],
        "industry": rng.choice(INDUSTRIES, size=N_ACCOUNTS),
        "customer_type": rng.choice(CUSTOMER_TYPES, size=N_ACCOUNTS, p=[0.7, 0.3]),
        "contract_length_months": rng.integers(6, 36, size=N_ACCOUNTS),
        "region": rng.choice(REGIONS, size=N_ACCOUNTS),
        "client_tenure_months": client_tenure
    })

    product_usage_rows = []
    survey_rows = []
    finance_rows = []

    churned_accounts = set()

    for period in MONTHS:
        base_date = period.end_time
        date_str = base_date.strftime("%Y-%m-%d")

        active_accounts = accounts[~accounts.account_id.isin(churned_accounts)]
        if active_accounts.empty:
            break

        # Generate usage
        engaged_mask = rng.random(len(active_accounts)) < 0.8

        sessions = np.where(
            engaged_mask,
            rng.normal(loc=14, scale=3, size=len(active_accounts)),
            rng.normal(loc=3, scale=2, size=len(active_accounts))
        )

        login_days = np.where(
            engaged_mask,
            rng.normal(loc=2, scale=1, size=len(active_accounts)),
            rng.normal(loc=25, scale=6, size=len(active_accounts))
        )

        support_tix = np.where(
            engaged_mask,
            rng.poisson(1, size=len(active_accounts)),
            rng.poisson(4, size=len(active_accounts))
        )

        usage_df = pd.DataFrame({
            "account_id": active_accounts.account_id.values,
            "date": date_str,
            "feature_purchased": rng.choice(FEATURES, size=len(active_accounts), p=[0.4, 0.2, 0.15, 0.15, 0.1]),
            "sessions_last_30d": np.clip(sessions, 0, None),
            "last_login_days": np.clip(login_days, 0, None),
            "support_tickets_30d": support_tix
        })
        product_usage_rows.append(usage_df)

        # Lookups for churn scoring
        usage_lookup = usage_df.set_index("account_id")

        if survey_rows:
            csat_subset = pd.DataFrame(survey_rows)
            csat_subset = csat_subset[pd.to_datetime(csat_subset["date"]) <= base_date]
            csat_lookup = (
                csat_subset
                .sort_values("date")
                .drop_duplicates("account_id", keep="last")
                .set_index("account_id")
            )
        else:
            csat_lookup = pd.DataFrame(columns=["csat_score"]).set_index(pd.Index([]))

        sessions = usage_lookup["sessions_last_30d"].values
        last_login = usage_lookup["last_login_days"].values
        support_tix = usage_lookup["support_tickets_30d"].values
        client_tenure_month = active_accounts["client_tenure_months"].values
        csat_month = csat_lookup.reindex(active_accounts.account_id).csat_score.fillna(5).values

        score = (
            -0.2 * sessions +
            0.2 * last_login +
            2.0 * support_tix -
            1.5 * (csat_month - 5) +
            1.5 * client_tenure_month +
            rng.normal(0, 0.5, len(active_accounts))
        )

        # ✅ Compute adjustment after score is defined
        remaining_ratio = len(active_accounts) / N_ACCOUNTS
        adjustment = 1.0 if remaining_ratio > 0.4 else 0.6 if remaining_ratio > 0.2 else 0.3
        churn_flag = rng.random(len(active_accounts)) < sigmoid(score - 2.5) * adjustment
        churned_ids = active_accounts.account_id[churn_flag].tolist()

        still_active_accounts = active_accounts[~active_accounts.account_id.isin(churned_ids)]

        for idx, acc in still_active_accounts.iterrows():
            engaged = rng.random() < 0.8
            base_csat = rng.normal(8.5, 0.6) if engaged else rng.normal(5.5, 1.2)
            csat_score = np.clip(base_csat, 1, 10)
            sentiment_score = np.clip((csat_score - 5) / 5 + rng.normal(0, 0.2), -1, 1)

            survey_rows.append({
                "account_id": acc.account_id,
                "date": (base_date - timedelta(days=int(rng.integers(0, 10)))).strftime("%Y-%m-%d"),
                "csat_score": round(csat_score, 1),
                "comment_sentiment_score": round(sentiment_score, 2),
            })

        churned_accounts.update(churned_ids)

        baseline_rev = rng.gamma(shape=2.2, scale=1200, size=len(active_accounts))
        closed_won = baseline_rev
        churned = closed_won * churn_flag
        actual = closed_won - churned

        finance_rows.append(pd.DataFrame({
            "account_id": active_accounts.account_id.values,
            "month": date_str,
            "closed_won_usd": closed_won.round(2),
            "churned_usd": churned.round(2),
            "actual_usd": actual.round(2),
            "churn": churn_flag.astype(int),
            "active_customer": (~churn_flag).astype(int),
        }))

    product_usage = pd.concat(product_usage_rows, ignore_index=True)
    aud_survey = pd.DataFrame(survey_rows)
    finance = pd.concat(finance_rows, ignore_index=True)

    for period in MONTHS:
        eom_str = period.end_time.strftime("%Y-%m-%d")
        mask = finance.month == eom_str
        total = finance.loc[mask, "actual_usd"].sum()
        factor = rng.uniform(1_000_000, 3_000_000) / (total if total else 1)
        finance.loc[mask, ["closed_won_usd", "churned_usd", "actual_usd"]] = (
            finance.loc[mask, ["closed_won_usd", "churned_usd", "actual_usd"]] * factor
        ).round(2)

    crm = accounts.copy()
    crm["deal_value_usd"] = rng.normal(15_000, 4000, N_ACCOUNTS).clip(2_000).round(2)
    crm["deal_status"] = rng.choice(
        DEAL_STATUS_VALUES, size=N_ACCOUNTS, p=[0.78, 0.13, 0.06, 0.03]
    )

    product_usage.to_csv("/opt/airflow/raw_data/product_usage.csv", index=False)
    crm.to_csv("/opt/airflow/raw_data/crm.csv", index=False, quoting=csv.QUOTE_ALL)
    finance.to_csv("/opt/airflow/raw_data/finance.csv", index=False)
    aud_survey.to_csv("/opt/airflow/raw_data/survey.csv", index=False)

    print("\u2705 Datasets generated:")
    for f in ("product_usage.csv", "crm.csv", "finance.csv", "survey.csv"):
        print(f" \u2022 {f}")

if __name__ == "__main__":
    data_generation()
