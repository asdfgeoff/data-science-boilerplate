SELECT COUNT(*)
FROM {schema_name}.{table_name}
WHERE {sort_key} BETWEEN '{date_from}' AND '{date_until}';