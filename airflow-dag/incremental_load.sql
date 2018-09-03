/* CONDITIONAL DDL */
CREATE TABLE IF NOT EXISTS {{ params.schema_name }}.{{ params.table_name}} (
  id              VARCHAR(36)   ENCODE ZSTD,
	dimension       VARCHAR(50)   ENCODE ZSTD,
	ts              TIMESTAMP     ENCODE ZSTD
)
  DISTSTYLE EVEN
  SORTKEY("ts");


/* STAGING TABLE */
DROP TABLE IF EXISTS stg_final;
CREATE TEMP TABLE stg_final AS (
  SELECT * FROM my_table
);


/* INCREMENTAL DELETE */
DELETE FROM {{ params.schema_name }}.{{ params.table_name}}
WHERE ts >= '{{ ds }}'
  AND ts < DATE('{{ ds }}') + interval '1 day';


/* INREMENTAL LOAD */
INSERT INTO {{ params.schema_name }}.{{ params.table_name}} (
  SELECT * FROM stg_final
);


/* PERMISSIONS */
ANALYZE {{ params.schema_name }}.{{ params.table_name}};
GRANT SELECT ON {{ params.schema_name }}.{{ params.table_name}} TO GROUP analyst;
GRANT ALL ON {{ params.schema_name }}.{{ params.table_name}} TO GROUP analytics_admin;