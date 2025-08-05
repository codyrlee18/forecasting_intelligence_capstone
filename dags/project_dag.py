from airflow.operators.dummy import DummyOperator
from airflow.decorators import dag, task
import pendulum
from datetime import timedelta
import sys
sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'Cody Lee',
    'start_date': pendulum.now(),
    'depends_on_past': False,
    'retries': 3,
    'retry_delay': timedelta(hours= 24),
}

@dag(
    default_args=default_args,
    description='Extract, Load, Transform, and Model data with Airflow',
    schedule_interval = timedelta(minutes=1000),
    catchup = False
)
def saas_dag():

    start_operator = DummyOperator(task_id='Begin_execution')

    @task 
    def data_generation():
        from scripts.data_gen import data_generation
        data_generation()

    @task
    def data_ingest():
        from scripts.ingestion import data_ingest
        data_ingest()

    @task
    def data_transform():
        from scripts.transform import data_transform
        data_transform()

    @task
    def churn_analysis():
        from scripts.churn_modeling import churn_analysis
        churn_analysis()

    @task
    def rev_analysis():
        from scripts.forecast_modeling import rev_analysis
        rev_analysis()

    end_operator = DummyOperator(task_id='Stop_execution')

    # flow of DAG

    t1 = data_generation()
    t2 = data_ingest()
    t3 = data_transform()
    t4 = churn_analysis()
    t5 = rev_analysis()

    start_operator >> t1 >> t2 >> t3 >> [t4, t5] >> end_operator

dag = saas_dag()