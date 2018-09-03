# Jupyter analytics boilerplate

## Purpose

Inspired by [http://drivendata.github.io/cookiecutter-data-science/](Cookiecutter Data Science), this directory represents a platonic ideal from which I begin new data analysis tasks.

### Principles
- By default everything in the `/data` directory is gitignored. Don't manually place data there, but rather include the code necessary to reproduce the data from scratch when run.
- A properly documented notebook has enoguh markdown that it can be used as a self-contained report when the code is hidden. When in doubt, hide your code and see if the notebook still makes sense.

## Structure
```
├── sql                         <- Templated SQL files
│   └── example_query.sql
├── data                        <- Cached data sets
│   ├── example_query.parquet
│   └── example_query.tsv
├── notebook.ipynb              <- Main analysis
└── README.md                   <- When completed, write a summary
```
