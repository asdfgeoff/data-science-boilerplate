from airflow.operators.python_operator import PythonOperator


class SQLTemplatedPythonOperator(PythonOperator):
    """ Extend PythonOperator to receive a templated SQL query and also to display it in the "Rendered Template" tab in Airflow's UI.

    This is very helpful for troubleshooting specific task instances, since you can copy a propertly formatted query directly from
    the web UI rather than copying the contents of "templates_dict" and parsing it manually.

    Args:
        sql (str): File path or query text containing jinja2 variables to be filled by airflow templating engine.
        python_callable (func): Access final sql text from inside using kwargs['templates_dict']['query']

    """
    template_ext = ('.sql',)
    template_fields = ('sql', 'templates_dict')
    ui_color = "#ea5651"

    def __init__(self, python_callable, sql, op_args=None, op_kwargs=None, provide_context=False,
                 templates_dict=None, templates_exts=None, *args, **kwargs):
        super(PythonOperator, self).__init__(*args, **kwargs)
        if not callable(python_callable):
            raise ValueError('`python_callable` param must be callable')
        self.python_callable = python_callable
        self.sql = sql
        self.op_args = op_args or []
        self.op_kwargs = op_kwargs or {}
        self.provide_context = provide_context
        if templates_dict:
            self.templates_dict = {**templates_dict, **{'sql': sql}}
        else:
            self.templates_dict = {'sql': sql}
        if templates_exts:
            self.template_ext = templates_exts


def example_func(*args, **kwargs):
    sql = kwargs['templates_dict']['sql']
    print('sql: {}'.format(sql))


example_task = SQLTemplatedPythonOperator(
    task_id='example_task',
    sql='example_query.sql',
    python_callable=example_func,
    provide_context=True)