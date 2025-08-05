import duckdb
import pandas as pd

def data_transform():

    con = duckdb.connect("/opt/airflow/forecastintelligence.duckdb")

    con.execute("DROP TABLE IF EXISTS master_summary_table;")
    con.execute(""" 
                CREATE TABLE IF NOT EXISTS master_summary_table AS
                SELECT 
                    f.account_id
                    , f.month
                    , f.actual_usd
                    , f.closed_won_usd
                    , f.churn
                    , f.active_customer
                    , s.csat_score
                    , s.comment_sentiment_score
                    , crm.account_name
                    , crm.industry
                    , crm.contract_length_months
                    , crm.region
                    , crm.client_tenure_months
                    , crm.deal_value_usd
                    , crm.deal_status
                    , pu.feature_purchased
                    , pu.sessions_last_30d
                    , pu.last_login_days
                    , pu.support_tickets_30d
                FROM finance f
                LEFT JOIN survey s
                    on f.account_id = s.account_id AND f.month = s.date
                LEFT JOIN product_usage pu
                    on f.account_id = pu.account_id AND f.month = pu.date
                LEFT JOIN crm 
                    on f.account_id = crm.account_id
                """)

    df = con.sql("SELECT * FROM master_summary_table").df()
    df['month_period'] = df['month'].dt.to_period('M')

    # Reduce to one row per account, make aggregations
    churn_df = df.groupby('account_id').agg({
        'account_name': 'last',
        'churn': 'last', 
        'industry': 'last',
        'closed_won_usd': 'sum',
        'client_tenure_months': 'max',  # tenure won't change monthly
        'deal_status': 'last',
        'feature_purchased': 'last',
        'sessions_last_30d': 'mean',  # avg usage
        'region': 'last',
        'csat_score': 'last',
        'comment_sentiment_score': 'last',
        'support_tickets_30d': 'mean',
        'last_login_days': 'mean'
    }).reset_index()

    churn_df.to_csv("/opt/airflow/clean_data/churn_model_data.csv", index=False)

    df.to_csv("/opt/airflow/clean_data/revenue_forecast_data.csv", index=False)

    con.close()

if __name__ == "__main__":
    data_transform()