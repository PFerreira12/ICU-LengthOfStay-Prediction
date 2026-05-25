-- BigQuery preprocessing with 9 tables:
-- 7-table version + OUTPUTEVENTS + PRESCRIPTIONS.

DECLARE window_hours INT64 DEFAULT {{window_hours}};
DECLARE top_n_chart_items INT64 DEFAULT {{top_n_chart_items}};
DECLARE top_n_lab_items INT64 DEFAULT {{top_n_lab_items}};
DECLARE top_n_output_items INT64 DEFAULT {{top_n_output_items}};
DECLARE top_n_drugs INT64 DEFAULT {{top_n_drugs}};
DECLARE chart_exprs STRING;
DECLARE lab_exprs STRING;
DECLARE output_exprs STRING;
DECLARE drug_exprs STRING;

CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.{{selected_items_table}}` AS
WITH icu AS (
  SELECT CAST(ICUSTAY_ID AS INT64) AS ICUSTAY_ID, SAFE_CAST(INTIME AS TIMESTAMP) AS INTIME
  FROM `{{project_id}}.{{dataset_id}}.{{icustays_table}}`
  WHERE ICUSTAY_ID IS NOT NULL AND INTIME IS NOT NULL
),
windowed_events AS (
  SELECT CAST(ce.ITEMID AS INT64) AS ITEMID
  FROM `{{project_id}}.{{dataset_id}}.{{chartevents_table}}` AS ce
  JOIN icu ON CAST(ce.ICUSTAY_ID AS INT64) = icu.ICUSTAY_ID
  WHERE ce.ITEMID IS NOT NULL
    AND SAFE_CAST(ce.VALUENUM AS FLOAT64) IS NOT NULL
    AND SAFE_CAST(ce.CHARTTIME AS TIMESTAMP) >= icu.INTIME
    AND SAFE_CAST(ce.CHARTTIME AS TIMESTAMP) <= TIMESTAMP_ADD(icu.INTIME, INTERVAL window_hours HOUR)
),
ranked_items AS (
  SELECT ITEMID, COUNT(*) AS event_count
  FROM windowed_events
  GROUP BY ITEMID
  ORDER BY event_count DESC
  LIMIT {{top_n_chart_items}}
)
SELECT ranked_items.ITEMID, ranked_items.event_count, d.LABEL, d.CATEGORY, d.UNITNAME
FROM ranked_items
LEFT JOIN `{{project_id}}.{{dataset_id}}.{{d_items_table}}` AS d
  ON ranked_items.ITEMID = CAST(d.ITEMID AS INT64);

CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.{{selected_lab_items_table}}` AS
WITH icu AS (
  SELECT CAST(HADM_ID AS INT64) AS HADM_ID, SAFE_CAST(INTIME AS TIMESTAMP) AS INTIME
  FROM `{{project_id}}.{{dataset_id}}.{{icustays_table}}`
  WHERE HADM_ID IS NOT NULL AND INTIME IS NOT NULL
),
windowed_labs AS (
  SELECT CAST(le.ITEMID AS INT64) AS ITEMID
  FROM `{{project_id}}.{{dataset_id}}.{{labevents_table}}` AS le
  JOIN icu ON CAST(le.HADM_ID AS INT64) = icu.HADM_ID
  WHERE le.ITEMID IS NOT NULL
    AND SAFE_CAST(le.VALUENUM AS FLOAT64) IS NOT NULL
    AND SAFE_CAST(le.CHARTTIME AS TIMESTAMP) >= icu.INTIME
    AND SAFE_CAST(le.CHARTTIME AS TIMESTAMP) <= TIMESTAMP_ADD(icu.INTIME, INTERVAL window_hours HOUR)
),
ranked_items AS (
  SELECT ITEMID, COUNT(*) AS event_count
  FROM windowed_labs
  GROUP BY ITEMID
  ORDER BY event_count DESC
  LIMIT {{top_n_lab_items}}
)
SELECT ranked_items.ITEMID, ranked_items.event_count, d.LABEL, d.FLUID, d.CATEGORY
FROM ranked_items
LEFT JOIN `{{project_id}}.{{dataset_id}}.{{d_labitems_table}}` AS d
  ON ranked_items.ITEMID = CAST(d.ITEMID AS INT64);

CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.{{selected_output_items_table}}` AS
WITH icu AS (
  SELECT CAST(ICUSTAY_ID AS INT64) AS ICUSTAY_ID, SAFE_CAST(INTIME AS TIMESTAMP) AS INTIME
  FROM `{{project_id}}.{{dataset_id}}.{{icustays_table}}`
  WHERE ICUSTAY_ID IS NOT NULL AND INTIME IS NOT NULL
),
windowed_outputs AS (
  SELECT CAST(oe.ITEMID AS INT64) AS ITEMID
  FROM `{{project_id}}.{{dataset_id}}.{{outputevents_table}}` AS oe
  JOIN icu ON CAST(oe.ICUSTAY_ID AS INT64) = icu.ICUSTAY_ID
  WHERE oe.ITEMID IS NOT NULL
    AND SAFE_CAST(oe.VALUE AS FLOAT64) IS NOT NULL
    AND SAFE_CAST(oe.CHARTTIME AS TIMESTAMP) >= icu.INTIME
    AND SAFE_CAST(oe.CHARTTIME AS TIMESTAMP) <= TIMESTAMP_ADD(icu.INTIME, INTERVAL window_hours HOUR)
),
ranked_items AS (
  SELECT ITEMID, COUNT(*) AS event_count
  FROM windowed_outputs
  GROUP BY ITEMID
  ORDER BY event_count DESC
  LIMIT {{top_n_output_items}}
)
SELECT ranked_items.ITEMID, ranked_items.event_count, d.LABEL, d.CATEGORY, d.UNITNAME
FROM ranked_items
LEFT JOIN `{{project_id}}.{{dataset_id}}.{{d_items_table}}` AS d
  ON ranked_items.ITEMID = CAST(d.ITEMID AS INT64);

CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.{{selected_drugs_table}}` AS
WITH icu AS (
  SELECT CAST(HADM_ID AS INT64) AS HADM_ID, SAFE_CAST(INTIME AS TIMESTAMP) AS INTIME
  FROM `{{project_id}}.{{dataset_id}}.{{icustays_table}}`
  WHERE HADM_ID IS NOT NULL AND INTIME IS NOT NULL
),
windowed_drugs AS (
  SELECT
    UPPER(TRIM(COALESCE(CAST(pr.DRUG AS STRING), CAST(pr.DRUG_NAME_GENERIC AS STRING), 'UNKNOWN'))) AS DRUG_NAME
  FROM `{{project_id}}.{{dataset_id}}.{{prescriptions_table}}` AS pr
  JOIN icu ON CAST(pr.HADM_ID AS INT64) = icu.HADM_ID
  WHERE COALESCE(pr.DRUG, pr.DRUG_NAME_GENERIC) IS NOT NULL
    AND SAFE_CAST(pr.STARTDATE AS TIMESTAMP) >= icu.INTIME
    AND SAFE_CAST(pr.STARTDATE AS TIMESTAMP) <= TIMESTAMP_ADD(icu.INTIME, INTERVAL window_hours HOUR)
),
ranked_drugs AS (
  SELECT
    SUBSTR(CONCAT('drug_', REGEXP_REPLACE(LOWER(DRUG_NAME), r'[^a-z0-9]+', '_')), 1, 100) AS DRUG_KEY,
    ARRAY_AGG(DISTINCT DRUG_NAME ORDER BY DRUG_NAME LIMIT 5) AS DRUG_NAMES,
    COUNT(*) AS prescription_count
  FROM windowed_drugs
  GROUP BY DRUG_KEY
  ORDER BY prescription_count DESC
  LIMIT {{top_n_drugs}}
)
SELECT * FROM ranked_drugs;

SET chart_exprs = (
  SELECT STRING_AGG(CONCAT(
    "MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", mean_value, NULL)) AS chart_", CAST(ITEMID AS STRING), "_mean,\n",
    "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", min_value, NULL)) AS chart_", CAST(ITEMID AS STRING), "_min,\n",
    "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", max_value, NULL)) AS chart_", CAST(ITEMID AS STRING), "_max,\n",
    "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", std_value, NULL)) AS chart_", CAST(ITEMID AS STRING), "_std,\n",
    "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", event_count, NULL)) AS chart_", CAST(ITEMID AS STRING), "_count,\n",
    "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", last_value, NULL)) AS chart_", CAST(ITEMID AS STRING), "_last"), ",\n")
  FROM `{{project_id}}.{{dataset_id}}.{{selected_items_table}}`
);

SET lab_exprs = (
  SELECT STRING_AGG(CONCAT(
    "MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", mean_value, NULL)) AS lab_", CAST(ITEMID AS STRING), "_mean,\n",
    "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", min_value, NULL)) AS lab_", CAST(ITEMID AS STRING), "_min,\n",
    "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", max_value, NULL)) AS lab_", CAST(ITEMID AS STRING), "_max,\n",
    "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", std_value, NULL)) AS lab_", CAST(ITEMID AS STRING), "_std,\n",
    "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", event_count, NULL)) AS lab_", CAST(ITEMID AS STRING), "_count,\n",
    "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", last_value, NULL)) AS lab_", CAST(ITEMID AS STRING), "_last"), ",\n")
  FROM `{{project_id}}.{{dataset_id}}.{{selected_lab_items_table}}`
);

SET output_exprs = (
  SELECT STRING_AGG(CONCAT(
    "MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", total_value, NULL)) AS output_", CAST(ITEMID AS STRING), "_sum,\n",
    "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", event_count, NULL)) AS output_", CAST(ITEMID AS STRING), "_count,\n",
    "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", last_value, NULL)) AS output_", CAST(ITEMID AS STRING), "_last"), ",\n")
  FROM `{{project_id}}.{{dataset_id}}.{{selected_output_items_table}}`
);

SET drug_exprs = (
  SELECT STRING_AGG(CONCAT(
    "MAX(IF(DRUG_KEY = '", DRUG_KEY, "', prescription_count, NULL)) AS ", DRUG_KEY, "_count"), ",\n")
  FROM `{{project_id}}.{{dataset_id}}.{{selected_drugs_table}}`
);

EXECUTE IMMEDIATE FORMAT("""
CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.{{features_table}}` AS
WITH icu AS (
  SELECT
    CAST(SUBJECT_ID AS INT64) AS SUBJECT_ID,
    CAST(HADM_ID AS INT64) AS HADM_ID,
    CAST(ICUSTAY_ID AS INT64) AS ICUSTAY_ID,
    UPPER(COALESCE(CAST(DBSOURCE AS STRING), 'UNKNOWN')) AS DBSOURCE,
    UPPER(COALESCE(CAST(FIRST_CAREUNIT AS STRING), 'UNKNOWN')) AS FIRST_CAREUNIT,
    UPPER(COALESCE(CAST(LAST_CAREUNIT AS STRING), 'UNKNOWN')) AS LAST_CAREUNIT,
    SAFE_CAST(INTIME AS TIMESTAMP) AS INTIME,
    COALESCE(
      SAFE_CAST(LOS AS FLOAT64),
      TIMESTAMP_DIFF(SAFE_CAST(OUTTIME AS TIMESTAMP), SAFE_CAST(INTIME AS TIMESTAMP), SECOND) / 86400.0
    ) AS LOS
  FROM `{{project_id}}.{{dataset_id}}.{{icustays_table}}`
  WHERE ICUSTAY_ID IS NOT NULL AND INTIME IS NOT NULL
),
patients AS (
  SELECT CAST(SUBJECT_ID AS INT64) AS SUBJECT_ID, UPPER(COALESCE(CAST(GENDER AS STRING), 'UNKNOWN')) AS GENDER, DATE(SAFE_CAST(DOB AS TIMESTAMP)) AS DOB
  FROM `{{project_id}}.{{dataset_id}}.{{patients_table}}`
),
admissions AS (
  SELECT
    CAST(SUBJECT_ID AS INT64) AS SUBJECT_ID,
    CAST(HADM_ID AS INT64) AS HADM_ID,
    UPPER(COALESCE(CAST(ADMISSION_TYPE AS STRING), 'UNKNOWN')) AS ADMISSION_TYPE,
    UPPER(COALESCE(CAST(ADMISSION_LOCATION AS STRING), 'UNKNOWN')) AS ADMISSION_LOCATION,
    UPPER(COALESCE(CAST(INSURANCE AS STRING), 'UNKNOWN')) AS INSURANCE,
    UPPER(COALESCE(CAST(LANGUAGE AS STRING), 'UNKNOWN')) AS LANGUAGE,
    UPPER(COALESCE(CAST(RELIGION AS STRING), 'UNKNOWN')) AS RELIGION,
    UPPER(COALESCE(CAST(MARITAL_STATUS AS STRING), 'UNKNOWN')) AS MARITAL_STATUS,
    UPPER(COALESCE(CAST(ETHNICITY AS STRING), 'UNKNOWN')) AS ETHNICITY,
    SAFE_CAST(HAS_CHARTEVENTS_DATA AS INT64) AS HAS_CHARTEVENTS_DATA,
    TIMESTAMP_DIFF(SAFE_CAST(EDOUTTIME AS TIMESTAMP), SAFE_CAST(EDREGTIME AS TIMESTAMP), SECOND) / 3600.0 AS ED_LOS_HOURS
  FROM `{{project_id}}.{{dataset_id}}.{{admissions_table}}`
),
chart_windowed AS (
  SELECT CAST(ce.ICUSTAY_ID AS INT64) AS ICUSTAY_ID, CAST(ce.ITEMID AS INT64) AS ITEMID, SAFE_CAST(ce.CHARTTIME AS TIMESTAMP) AS EVENT_TIME, SAFE_CAST(ce.VALUENUM AS FLOAT64) AS VALUE
  FROM `{{project_id}}.{{dataset_id}}.{{chartevents_table}}` AS ce
  JOIN icu ON CAST(ce.ICUSTAY_ID AS INT64) = icu.ICUSTAY_ID
  JOIN `{{project_id}}.{{dataset_id}}.{{selected_items_table}}` AS selected ON CAST(ce.ITEMID AS INT64) = selected.ITEMID
  WHERE SAFE_CAST(ce.VALUENUM AS FLOAT64) IS NOT NULL
    AND SAFE_CAST(ce.CHARTTIME AS TIMESTAMP) >= icu.INTIME
    AND SAFE_CAST(ce.CHARTTIME AS TIMESTAMP) <= TIMESTAMP_ADD(icu.INTIME, INTERVAL {{window_hours}} HOUR)
),
chart_aggregated AS (
  SELECT ICUSTAY_ID, ITEMID, AVG(VALUE) AS mean_value, MIN(VALUE) AS min_value, MAX(VALUE) AS max_value, STDDEV_SAMP(VALUE) AS std_value, COUNT(*) AS event_count, ARRAY_AGG(VALUE ORDER BY EVENT_TIME DESC LIMIT 1)[OFFSET(0)] AS last_value
  FROM chart_windowed
  GROUP BY ICUSTAY_ID, ITEMID
),
chart_pivot AS (
  SELECT ICUSTAY_ID, %s FROM chart_aggregated GROUP BY ICUSTAY_ID
),
lab_windowed AS (
  SELECT icu.ICUSTAY_ID, CAST(le.ITEMID AS INT64) AS ITEMID, SAFE_CAST(le.CHARTTIME AS TIMESTAMP) AS EVENT_TIME, SAFE_CAST(le.VALUENUM AS FLOAT64) AS VALUE
  FROM `{{project_id}}.{{dataset_id}}.{{labevents_table}}` AS le
  JOIN icu ON CAST(le.HADM_ID AS INT64) = icu.HADM_ID
  JOIN `{{project_id}}.{{dataset_id}}.{{selected_lab_items_table}}` AS selected ON CAST(le.ITEMID AS INT64) = selected.ITEMID
  WHERE SAFE_CAST(le.VALUENUM AS FLOAT64) IS NOT NULL
    AND SAFE_CAST(le.CHARTTIME AS TIMESTAMP) >= icu.INTIME
    AND SAFE_CAST(le.CHARTTIME AS TIMESTAMP) <= TIMESTAMP_ADD(icu.INTIME, INTERVAL {{window_hours}} HOUR)
),
lab_aggregated AS (
  SELECT ICUSTAY_ID, ITEMID, AVG(VALUE) AS mean_value, MIN(VALUE) AS min_value, MAX(VALUE) AS max_value, STDDEV_SAMP(VALUE) AS std_value, COUNT(*) AS event_count, ARRAY_AGG(VALUE ORDER BY EVENT_TIME DESC LIMIT 1)[OFFSET(0)] AS last_value
  FROM lab_windowed
  GROUP BY ICUSTAY_ID, ITEMID
),
lab_pivot AS (
  SELECT ICUSTAY_ID, %s FROM lab_aggregated GROUP BY ICUSTAY_ID
),
output_windowed AS (
  SELECT CAST(oe.ICUSTAY_ID AS INT64) AS ICUSTAY_ID, CAST(oe.ITEMID AS INT64) AS ITEMID, SAFE_CAST(oe.CHARTTIME AS TIMESTAMP) AS EVENT_TIME, SAFE_CAST(oe.VALUE AS FLOAT64) AS VALUE
  FROM `{{project_id}}.{{dataset_id}}.{{outputevents_table}}` AS oe
  JOIN icu ON CAST(oe.ICUSTAY_ID AS INT64) = icu.ICUSTAY_ID
  JOIN `{{project_id}}.{{dataset_id}}.{{selected_output_items_table}}` AS selected ON CAST(oe.ITEMID AS INT64) = selected.ITEMID
  WHERE SAFE_CAST(oe.VALUE AS FLOAT64) IS NOT NULL
    AND SAFE_CAST(oe.CHARTTIME AS TIMESTAMP) >= icu.INTIME
    AND SAFE_CAST(oe.CHARTTIME AS TIMESTAMP) <= TIMESTAMP_ADD(icu.INTIME, INTERVAL {{window_hours}} HOUR)
),
output_aggregated AS (
  SELECT ICUSTAY_ID, ITEMID, SUM(VALUE) AS total_value, COUNT(*) AS event_count, ARRAY_AGG(VALUE ORDER BY EVENT_TIME DESC LIMIT 1)[OFFSET(0)] AS last_value
  FROM output_windowed
  GROUP BY ICUSTAY_ID, ITEMID
),
output_pivot AS (
  SELECT ICUSTAY_ID, %s FROM output_aggregated GROUP BY ICUSTAY_ID
),
drug_windowed AS (
  SELECT icu.ICUSTAY_ID, selected.DRUG_KEY, COUNT(*) AS prescription_count
  FROM `{{project_id}}.{{dataset_id}}.{{prescriptions_table}}` AS pr
  JOIN icu ON CAST(pr.HADM_ID AS INT64) = icu.HADM_ID
  JOIN `{{project_id}}.{{dataset_id}}.{{selected_drugs_table}}` AS selected
    ON SUBSTR(CONCAT('drug_', REGEXP_REPLACE(LOWER(TRIM(COALESCE(CAST(pr.DRUG AS STRING), CAST(pr.DRUG_NAME_GENERIC AS STRING), 'UNKNOWN'))), r'[^a-z0-9]+', '_')), 1, 100) = selected.DRUG_KEY
  WHERE SAFE_CAST(pr.STARTDATE AS TIMESTAMP) >= icu.INTIME
    AND SAFE_CAST(pr.STARTDATE AS TIMESTAMP) <= TIMESTAMP_ADD(icu.INTIME, INTERVAL {{window_hours}} HOUR)
  GROUP BY icu.ICUSTAY_ID, selected.DRUG_KEY
),
drug_pivot AS (
  SELECT ICUSTAY_ID, %s FROM drug_windowed GROUP BY ICUSTAY_ID
)
SELECT
  icu.SUBJECT_ID,
  icu.HADM_ID,
  icu.ICUSTAY_ID,
  icu.DBSOURCE,
  icu.FIRST_CAREUNIT,
  icu.LAST_CAREUNIT,
  icu.LOS,
  patients.GENDER,
  CASE
    WHEN DATE_DIFF(DATE(icu.INTIME), patients.DOB, YEAR) > 120 THEN 90
    WHEN DATE_DIFF(DATE(icu.INTIME), patients.DOB, YEAR) < 0 THEN NULL
    ELSE DATE_DIFF(DATE(icu.INTIME), patients.DOB, YEAR)
  END AS AGE,
  admissions.ADMISSION_TYPE,
  admissions.ADMISSION_LOCATION,
  admissions.INSURANCE,
  admissions.LANGUAGE,
  admissions.RELIGION,
  admissions.MARITAL_STATUS,
  admissions.ETHNICITY,
  admissions.HAS_CHARTEVENTS_DATA,
  admissions.ED_LOS_HOURS,
  chart_pivot.* EXCEPT (ICUSTAY_ID),
  lab_pivot.* EXCEPT (ICUSTAY_ID),
  output_pivot.* EXCEPT (ICUSTAY_ID),
  drug_pivot.* EXCEPT (ICUSTAY_ID)
FROM icu
LEFT JOIN patients USING (SUBJECT_ID)
LEFT JOIN admissions USING (SUBJECT_ID, HADM_ID)
LEFT JOIN chart_pivot USING (ICUSTAY_ID)
LEFT JOIN lab_pivot USING (ICUSTAY_ID)
LEFT JOIN output_pivot USING (ICUSTAY_ID)
LEFT JOIN drug_pivot USING (ICUSTAY_ID)
WHERE icu.LOS IS NOT NULL
  AND chart_pivot.ICUSTAY_ID IS NOT NULL
""", chart_exprs, lab_exprs, output_exprs, drug_exprs);

CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.{{quality_report_table}}` AS
WITH metrics AS (
  SELECT 'version_tables' AS metric, 9.0 AS value
  UNION ALL SELECT 'window_hours', CAST(window_hours AS FLOAT64)
  UNION ALL SELECT 'selected_chart_items', CAST(COUNT(*) AS FLOAT64)
    FROM `{{project_id}}.{{dataset_id}}.{{selected_items_table}}`
  UNION ALL SELECT 'selected_lab_items', CAST(COUNT(*) AS FLOAT64)
    FROM `{{project_id}}.{{dataset_id}}.{{selected_lab_items_table}}`
  UNION ALL SELECT 'selected_output_items', CAST(COUNT(*) AS FLOAT64)
    FROM `{{project_id}}.{{dataset_id}}.{{selected_output_items_table}}`
  UNION ALL SELECT 'selected_drugs', CAST(COUNT(*) AS FLOAT64)
    FROM `{{project_id}}.{{dataset_id}}.{{selected_drugs_table}}`
  UNION ALL SELECT 'preprocessed_rows', CAST(COUNT(*) AS FLOAT64)
    FROM `{{project_id}}.{{dataset_id}}.{{features_table}}`
)
SELECT * FROM metrics;
