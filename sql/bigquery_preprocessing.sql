-- BigQuery preprocessing for ICU length-of-stay prediction.
-- Template variables are rendered by src/preprocessing/bigquery_preprocess.py.

DECLARE window_hours INT64 DEFAULT {{window_hours}};
DECLARE top_n_items INT64 DEFAULT {{top_n_items}};
DECLARE feature_exprs STRING;

CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.{{selected_items_table}}` AS
WITH icu AS (
  SELECT
    CAST(SUBJECT_ID AS INT64) AS SUBJECT_ID,
    CAST(HADM_ID AS INT64) AS HADM_ID,
    CAST(ICUSTAY_ID AS INT64) AS ICUSTAY_ID,
    SAFE_CAST(INTIME AS TIMESTAMP) AS INTIME,
    COALESCE(
      SAFE_CAST(LOS AS FLOAT64),
      TIMESTAMP_DIFF(SAFE_CAST(OUTTIME AS TIMESTAMP), SAFE_CAST(INTIME AS TIMESTAMP), SECOND) / 86400.0
    ) AS LOS
  FROM `{{project_id}}.{{dataset_id}}.{{icustays_table}}`
  WHERE ICUSTAY_ID IS NOT NULL
    AND INTIME IS NOT NULL
),
windowed_events AS (
  SELECT
    CAST(ce.ICUSTAY_ID AS INT64) AS ICUSTAY_ID,
    CAST(ce.ITEMID AS INT64) AS ITEMID,
    SAFE_CAST(ce.CHARTTIME AS TIMESTAMP) AS CHARTTIME,
    SAFE_CAST(ce.VALUENUM AS FLOAT64) AS VALUENUM
  FROM `{{project_id}}.{{dataset_id}}.{{chartevents_table}}` AS ce
  JOIN icu USING (ICUSTAY_ID)
  WHERE ce.ICUSTAY_ID IS NOT NULL
    AND ce.ITEMID IS NOT NULL
    AND ce.CHARTTIME IS NOT NULL
    AND SAFE_CAST(ce.VALUENUM AS FLOAT64) IS NOT NULL
    AND SAFE_CAST(ce.CHARTTIME AS TIMESTAMP) >= icu.INTIME
    AND SAFE_CAST(ce.CHARTTIME AS TIMESTAMP) <= TIMESTAMP_ADD(icu.INTIME, INTERVAL window_hours HOUR)
),
ranked_items AS (
  SELECT
    ITEMID,
    COUNT(*) AS event_count
  FROM windowed_events
  GROUP BY ITEMID
  ORDER BY event_count DESC
  LIMIT {{top_n_items}}
)
SELECT
  ranked_items.ITEMID,
  ranked_items.event_count,
  d.LABEL,
  d.CATEGORY,
  d.UNITNAME
FROM ranked_items
LEFT JOIN `{{project_id}}.{{dataset_id}}.{{d_items_table}}` AS d
  ON ranked_items.ITEMID = CAST(d.ITEMID AS INT64);

SET feature_exprs = (
  SELECT STRING_AGG(
    CONCAT(
      "MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", mean_value, NULL)) AS item_", CAST(ITEMID AS STRING), "_mean,\n",
      "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", min_value, NULL)) AS item_", CAST(ITEMID AS STRING), "_min,\n",
      "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", max_value, NULL)) AS item_", CAST(ITEMID AS STRING), "_max,\n",
      "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", std_value, NULL)) AS item_", CAST(ITEMID AS STRING), "_std,\n",
      "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", event_count, NULL)) AS item_", CAST(ITEMID AS STRING), "_count,\n",
      "      MAX(IF(ITEMID = ", CAST(ITEMID AS STRING), ", last_value, NULL)) AS item_", CAST(ITEMID AS STRING), "_last"
    ),
    ",\n"
  )
  FROM `{{project_id}}.{{dataset_id}}.{{selected_items_table}}`
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
  WHERE ICUSTAY_ID IS NOT NULL
    AND INTIME IS NOT NULL
),
patients AS (
  SELECT
    CAST(SUBJECT_ID AS INT64) AS SUBJECT_ID,
    UPPER(COALESCE(CAST(GENDER AS STRING), 'UNKNOWN')) AS GENDER,
    DATE(SAFE_CAST(DOB AS TIMESTAMP)) AS DOB
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
windowed_events AS (
  SELECT
    CAST(ce.ICUSTAY_ID AS INT64) AS ICUSTAY_ID,
    CAST(ce.ITEMID AS INT64) AS ITEMID,
    SAFE_CAST(ce.CHARTTIME AS TIMESTAMP) AS CHARTTIME,
    SAFE_CAST(ce.VALUENUM AS FLOAT64) AS VALUENUM
  FROM `{{project_id}}.{{dataset_id}}.{{chartevents_table}}` AS ce
  JOIN icu USING (ICUSTAY_ID)
  JOIN `{{project_id}}.{{dataset_id}}.{{selected_items_table}}` AS selected
    ON CAST(ce.ITEMID AS INT64) = selected.ITEMID
  WHERE ce.ICUSTAY_ID IS NOT NULL
    AND ce.ITEMID IS NOT NULL
    AND ce.CHARTTIME IS NOT NULL
    AND SAFE_CAST(ce.VALUENUM AS FLOAT64) IS NOT NULL
    AND SAFE_CAST(ce.CHARTTIME AS TIMESTAMP) >= icu.INTIME
    AND SAFE_CAST(ce.CHARTTIME AS TIMESTAMP) <= TIMESTAMP_ADD(icu.INTIME, INTERVAL {{window_hours}} HOUR)
),
aggregated AS (
  SELECT
    ICUSTAY_ID,
    ITEMID,
    AVG(VALUENUM) AS mean_value,
    MIN(VALUENUM) AS min_value,
    MAX(VALUENUM) AS max_value,
    STDDEV_SAMP(VALUENUM) AS std_value,
    COUNT(*) AS event_count,
    ARRAY_AGG(VALUENUM ORDER BY CHARTTIME DESC LIMIT 1)[OFFSET(0)] AS last_value
  FROM windowed_events
  GROUP BY ICUSTAY_ID, ITEMID
),
pivoted AS (
  SELECT
    ICUSTAY_ID,
    %s
  FROM aggregated
  GROUP BY ICUSTAY_ID
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
  pivoted.* EXCEPT (ICUSTAY_ID)
FROM icu
JOIN pivoted USING (ICUSTAY_ID)
LEFT JOIN patients USING (SUBJECT_ID)
LEFT JOIN admissions USING (SUBJECT_ID, HADM_ID)
WHERE icu.LOS IS NOT NULL
""", feature_exprs);

CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.{{quality_report_table}}` AS
WITH icu AS (
  SELECT
    CAST(SUBJECT_ID AS INT64) AS SUBJECT_ID,
    CAST(ICUSTAY_ID AS INT64) AS ICUSTAY_ID,
    SAFE_CAST(INTIME AS TIMESTAMP) AS INTIME,
    COALESCE(
      SAFE_CAST(LOS AS FLOAT64),
      TIMESTAMP_DIFF(SAFE_CAST(OUTTIME AS TIMESTAMP), SAFE_CAST(INTIME AS TIMESTAMP), SECOND) / 86400.0
    ) AS LOS
  FROM `{{project_id}}.{{dataset_id}}.{{icustays_table}}`
  WHERE ICUSTAY_ID IS NOT NULL
    AND INTIME IS NOT NULL
),
valid_events AS (
  SELECT
    CAST(ce.ICUSTAY_ID AS INT64) AS ICUSTAY_ID,
    SAFE_CAST(ce.CHARTTIME AS TIMESTAMP) AS CHARTTIME,
    SAFE_CAST(ce.VALUENUM AS FLOAT64) AS VALUENUM
  FROM `{{project_id}}.{{dataset_id}}.{{chartevents_table}}` AS ce
  WHERE ce.ICUSTAY_ID IS NOT NULL
    AND ce.CHARTTIME IS NOT NULL
    AND SAFE_CAST(ce.VALUENUM AS FLOAT64) IS NOT NULL
),
linked AS (
  SELECT
    valid_events.*,
    icu.INTIME
  FROM valid_events
  JOIN icu USING (ICUSTAY_ID)
),
metrics AS (
  SELECT 'window_hours' AS metric, CAST(window_hours AS FLOAT64) AS value
  UNION ALL SELECT 'top_n_items', CAST(top_n_items AS FLOAT64)
  UNION ALL SELECT 'selected_items', CAST(COUNT(*) AS FLOAT64)
    FROM `{{project_id}}.{{dataset_id}}.{{selected_items_table}}`
  UNION ALL SELECT 'icustays_after_target_filter', CAST(COUNT(*) AS FLOAT64)
    FROM icu
    WHERE LOS IS NOT NULL
  UNION ALL SELECT 'patients_after_target_filter', CAST(COUNT(DISTINCT SUBJECT_ID) AS FLOAT64)
    FROM icu
    WHERE LOS IS NOT NULL
  UNION ALL SELECT 'preprocessed_rows', CAST(COUNT(*) AS FLOAT64)
    FROM `{{project_id}}.{{dataset_id}}.{{features_table}}`
  UNION ALL SELECT 'valid_numeric_chartevents', CAST(COUNT(*) AS FLOAT64)
    FROM valid_events
  UNION ALL SELECT 'chartevents_linked_to_icustay', CAST(COUNT(*) AS FLOAT64)
    FROM linked
  UNION ALL SELECT 'chartevents_before_intime', CAST(COUNTIF(CHARTTIME < INTIME) AS FLOAT64)
    FROM linked
  UNION ALL SELECT 'chartevents_in_prediction_window', CAST(COUNTIF(CHARTTIME >= INTIME AND CHARTTIME <= TIMESTAMP_ADD(INTIME, INTERVAL window_hours HOUR)) AS FLOAT64)
    FROM linked
  UNION ALL SELECT 'chartevents_after_prediction_window', CAST(COUNTIF(CHARTTIME > TIMESTAMP_ADD(INTIME, INTERVAL window_hours HOUR)) AS FLOAT64)
    FROM linked
)
SELECT * FROM metrics;
