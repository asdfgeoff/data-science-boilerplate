WITH duplicates AS (
  SELECT
    {% set comma = joiner(", ") -%}
    {%- for column in cols -%}
    {{ comma() }}{{ column }}
    {%- endfor -%}
    , COUNT(*) AS num_rows
  FROM {{ schema }}.{{ table }}
  WHERE
    {%- set and = joiner("AND ") %}
    {%- for column in cols %}
    {{ and() }}{{ column }} IS NOT NULL
    {%- endfor %}
  GROUP BY
    {% set comma = joiner(", ") -%}
    {%- for column in cols -%}
    {{ comma() }}{{ column }}
    {%- endfor %}
  HAVING COUNT(*) > 1
  )
SELECT NVL(SUM(num_rows), 0) AS num_duplicates
FROM duplicates
;