import os
from datetime import datetime, timedelta

DAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dag_datasci_enriched_bookings')
print('DAG DIR: {}'.format(DAG_DIR))

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
filename = os.path.splitext(os.path.basename(__file__))[0]

def first_func():
    pass


def second_func():
    pass


with DAG('my_neat_dag',
         schedule_interval=timedelta(1),
         depends_on_past=False,
         start_date=datetime(2017, 1, 1),
         autocommit=True,
         template_searchpath=DAG_DIR) as dag:

    task_1 = PythonOperator(
        task_id='task_1',
        python_callable=first_func)

    task_2 = PythonOperator(
        task_id='task_2',
        python_callable=second_func)

    task_1 >> task_2
