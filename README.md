# Data Science Boilerplate

## Purpose

> Data scientists spend 80% of their time preparing and cleaning their data.
They spend the other 20% of their time complaining about preparing and cleaning their data.

– [@KirkDBorne](https://twitter.com/KirkDBorne/status/950471742228713472) (Twitter)

It is a common joke that data scientists spend an outsized proportion of their time performing repetitive work to prepare, clean and transform their data compared to model tuning and refinement.

This repository contains some boilerplate code, functions, and notebooks which I have abstracted out to be reused across projects. Perhaps you'll find something useful.



## Structure

`airflow-dag` – Template for a basic DAG in Apache Airflow which performs a bunch of server-side SQL tasks.

`jupyter-notebook` – Template for a jupyter notebook to perform some sort of ad-hoc analysis.

`machine-learning` – Some handy utility functions for doing basic ML work, and some notebooks to act as starting points for approaching similar problems in the future.