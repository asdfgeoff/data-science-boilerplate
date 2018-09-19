SELECT COUNT(*)
FROM {{ schema }}.{{ table }}
WHERE {{ column }} IS NULL