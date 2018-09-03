import os
import pandas as pd
from jinja2 import Template
from airflow import AirflowException
from airflow.hooks.base_hook import BaseHook
from airflow.operators.python_operator import PythonOperator

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
conn = BaseHook.get_connection('name_of_airflow_connection')
cx_str = 'postgresql://{db_user}:{db_pass}@{host}:{port}/{db}'.format(db_user=conn.login,
                                                                      db_pass=conn.get_password(),
                                                                      host=conn.host,
                                                                      port=conn.port,
                                                                      db=conn.schema)


def discover_columns(schema, table):
    with open(os.path.join(LOCAL_DIR, 'query_templates', 'discover_columns.sql')) as f:
        query = Template(f.read()).render(schema=schema, table=table)
    results = pd.read_sql(query, cx_str).iloc[:, 0].values
    if len(results) == 0:
        raise AirflowException('Cound not discover columns for table {}.{}'.format(schema, table))
    return results


def assert_no_nulls(schema, table, subset=None):
    if not subset:
        subset = discover_columns(schema, table)
        print('ASSERT NO NULLS IN ALL COLUMNS: {}'.format(subset))
    else:
        print('ASSERT NO NULLS IN SPECIFIC COLUMNS: {}'.format(subset))

    with open(os.path.join(LOCAL_DIR, 'query_templates', 'count_nulls.sql')) as f:
        template = Template(f.read())

    problem_cols = []

    for col in subset:
        query = template.render(schema=schema, table=table, column=col)
        result = pd.read_sql(query, cx_str).iloc[0, 0]
        if result > 0:
            problem_cols.append(col)

    if len(problem_cols) == 0:
        print('Success! All columns are non-null: {}'.format(', '.join(subset)))
    else:
        raise AirflowException('Uh-oh! These columns have null values: {}'.format(', '.join(problem_cols)))


def assert_unique_values(schema, table, subset=None):
    with open(os.path.join(LOCAL_DIR, 'query_templates', 'count_duplicates.sql')) as f:
        template = Template(f.read())

    if not subset:
        subset = discover_columns(schema, table)
    elif isinstance(subset, str):
        subset = [subset]

    print('ASSERT UNQUE VALUES IN SUBSET: {}'.format(subset))
    query = template.render(schema=schema, table=table, cols=subset)
    result = pd.read_sql(query, cx_str).iloc[0, 0]

    if result == 0:
        print('Success! All rows are unique across columns: {}'.format(', '.join(subset)))
    else:
        raise AirflowException('Uh-oh! Some rows are duplicated across columns {}'.format(', '.join(subset)))


def run_checks(schema, table, no_nulls=False, unique_rows=False, unique_subset=False):
    print('RUNNING CHECKS ON on {}.{}'.format(schema, table))
    print('Check for nulls? {}'.format(str(no_nulls).upper()))
    print('Check for unique rows? {}'.format(str(unique_rows).upper()))
    print('Check for unique subsets? {}'.format(str(unique_subset).upper()))

    if no_nulls is True:
        assert_no_nulls(schema, table)
    elif isinstance(no_nulls, list):
        assert_no_nulls(schema, table, subset=no_nulls)

    if unique_rows:
        assert_unique_values(schema, table)

    if unique_subset:
        assert_unique_values(schema, table, subset=unique_subset)


def throw_airflow_exception():
    raise AirflowException('INTENTIONAL FAIL')


class DataQualityOperator(PythonOperator):
    """ """

    ui_color = "#4c8244"
    ui_fgcolor = "#fff"

    def __init__(self, schema, table, no_nulls=False, unique_rows=False, unique_subset=False, *args, **kwargs):
        super(PythonOperator, self).__init__(python_callable=run_checks, *args, **kwargs)
        self.python_callable = run_checks
        self.op_args = [schema, table]
        self.op_kwargs = dict(no_nulls=no_nulls, unique_rows=unique_rows, unique_subset=unique_subset)
        self.provide_context = False
        self.templates_dict = None


# example_task = DataQualityOperator(
#     task_id='example_task',
#     schema='superb_schema',
#     table='terrific_table',
#     no_nulls=True,
#     unique_rows=True,
#     unique_subsets=['session_id'],
#     provide_context=True)
