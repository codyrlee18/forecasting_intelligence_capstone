import duckdb
import pandas


def data_ingest():
    
   con = duckdb.connect(database = 'forecastintelligence.duckdb', read_only=False)

   con.execute("DROP TABLE IF EXISTS survey;")
   con.execute(""" 
               CREATE TABLE IF NOT EXISTS survey (
                  account_id VARCHAR,
                  date DATE,
                  csat_score DOUBLE,
                  comment_sentiment_score DOUBLE
               );
               """)

   con.execute(""" 
               COPY survey
               FROM '/opt/airflow/raw_data/survey.csv'
               (AUTO_DETECT TRUE, HEADER TRUE);
               """)

   con.execute("DROP TABLE IF EXISTS finance;")
   con.execute(""" 
               CREATE TABLE IF NOT EXISTS finance (
               account_id VARCHAR,
               month DATE,
               closed_won_usd DOUBLE,
               churned_usd DOUBLE,
               actual_usd DOUBLE,
               churn BOOLEAN,
               active_customer BOOLEAN)
               """)

   con.execute("""
               COPY finance
               FROM '/opt/airflow/raw_data/finance.csv'
               (AUTO_DETECT TRUE, HEADER TRUE);
               """)

   con.execute("DROP TABLE IF EXISTS crm;")
   con.execute(""" 
               CREATE TABLE IF NOT EXISTS crm (
               account_id VARCHAR,
               account_name VARCHAR,
               industry VARCHAR,
               customer_type VARCHAR,
               contract_length_months DOUBLE,
               region VARCHAR,
               client_tenure_months DOUBLE,
               deal_value_usd DOUBLE,
               deal_status STRING)
               """)

   con.execute("""
               COPY crm
               FROM '/opt/airflow/raw_data/crm.csv'
               (AUTO_DETECT TRUE, HEADER TRUE);
               """)

   con.execute("DROP TABLE IF EXISTS product_usage;")
   con.execute(""" 
               CREATE TABLE IF NOT EXISTS product_usage (
               account_id VARCHAR,
               date DATE,
               feature_purchased VARCHAR,
               sessions_last_30d DOUBLE,
               last_login_days DOUBLE,
               support_tickets_30d DOUBLE)
               """)

   con.execute("""
               COPY product_usage
               FROM '/opt/airflow/raw_data/product_usage.csv'
               (AUTO_DETECT TRUE, HEADER TRUE);
               """)


   con.close()

if __name__ == "__main__":
    data_ingest()