docs = """
## DAG Name

#### Purpose

This DAG connects data from one source to another,
performs necessary transformations,
and creates a set of tables that can be used by analysts 

#### Outputs

This pipeline produces the following output tables:

- `table_A` – Contains useful information about ABC.
- `table_b` – Contains useful inormation about XYZ.

#### Owner

For any questions or concerns, please contact [me@mycompany.com](mailto:me@mycompany.com).
"""

import os
from datetime import datetime, timedelta

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

    dag.doc_md = docs

    task_1 = PythonOperator(
        task_id='task_1',
        python_callable=first_func)

    task_2 = PythonOperator(
        task_id='task_2',
        python_callable=second_func)

    task_1 >> task_2
