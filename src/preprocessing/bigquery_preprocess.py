from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from google.api_core import exceptions
from google.cloud import bigquery
from google.cloud import storage


ROOT = Path(__file__).resolve().parents[2]
SQL_TEMPLATE = ROOT / "sql" / "bigquery_preprocessing.sql"


TABLE_SCHEMAS = {
    "chartevents": [
        bigquery.SchemaField("ROW_ID", "INT64"),
        bigquery.SchemaField("SUBJECT_ID", "INT64"),
        bigquery.SchemaField("HADM_ID", "INT64"),
        bigquery.SchemaField("ICUSTAY_ID", "INT64"),
        bigquery.SchemaField("ITEMID", "INT64"),
        bigquery.SchemaField("CHARTTIME", "TIMESTAMP"),
        bigquery.SchemaField("STORETIME", "TIMESTAMP"),
        bigquery.SchemaField("CGID", "INT64"),
        bigquery.SchemaField("VALUE", "STRING"),
        bigquery.SchemaField("VALUENUM", "FLOAT64"),
        bigquery.SchemaField("VALUEUOM", "STRING"),
        bigquery.SchemaField("WARNING", "INT64"),
        bigquery.SchemaField("ERROR", "INT64"),
        bigquery.SchemaField("RESULTSTATUS", "STRING"),
        bigquery.SchemaField("STOPPED", "STRING"),
    ],
    "icustays": [
        bigquery.SchemaField("ROW_ID", "INT64"),
        bigquery.SchemaField("SUBJECT_ID", "INT64"),
        bigquery.SchemaField("HADM_ID", "INT64"),
        bigquery.SchemaField("ICUSTAY_ID", "INT64"),
        bigquery.SchemaField("DBSOURCE", "STRING"),
        bigquery.SchemaField("FIRST_CAREUNIT", "STRING"),
        bigquery.SchemaField("LAST_CAREUNIT", "STRING"),
        bigquery.SchemaField("FIRST_WARDID", "INT64"),
        bigquery.SchemaField("LAST_WARDID", "INT64"),
        bigquery.SchemaField("INTIME", "TIMESTAMP"),
        bigquery.SchemaField("OUTTIME", "TIMESTAMP"),
        bigquery.SchemaField("LOS", "FLOAT64"),
    ],
    "d_items": [
        bigquery.SchemaField("ROW_ID", "INT64"),
        bigquery.SchemaField("ITEMID", "INT64"),
        bigquery.SchemaField("LABEL", "STRING"),
        bigquery.SchemaField("ABBREVIATION", "STRING"),
        bigquery.SchemaField("DBSOURCE", "STRING"),
        bigquery.SchemaField("LINKSTO", "STRING"),
        bigquery.SchemaField("CATEGORY", "STRING"),
        bigquery.SchemaField("UNITNAME", "STRING"),
        bigquery.SchemaField("PARAM_TYPE", "STRING"),
        bigquery.SchemaField("CONCEPTID", "STRING"),
    ],
    "patients": [
        bigquery.SchemaField("ROW_ID", "INT64"),
        bigquery.SchemaField("SUBJECT_ID", "INT64"),
        bigquery.SchemaField("GENDER", "STRING"),
        bigquery.SchemaField("DOB", "TIMESTAMP"),
        bigquery.SchemaField("DOD", "TIMESTAMP"),
        bigquery.SchemaField("DOD_HOSP", "TIMESTAMP"),
        bigquery.SchemaField("DOD_SSN", "TIMESTAMP"),
        bigquery.SchemaField("EXPIRE_FLAG", "INT64"),
    ],
    "admissions": [
        bigquery.SchemaField("ROW_ID", "INT64"),
        bigquery.SchemaField("SUBJECT_ID", "INT64"),
        bigquery.SchemaField("HADM_ID", "INT64"),
        bigquery.SchemaField("ADMITTIME", "TIMESTAMP"),
        bigquery.SchemaField("DISCHTIME", "TIMESTAMP"),
        bigquery.SchemaField("DEATHTIME", "TIMESTAMP"),
        bigquery.SchemaField("ADMISSION_TYPE", "STRING"),
        bigquery.SchemaField("ADMISSION_LOCATION", "STRING"),
        bigquery.SchemaField("DISCHARGE_LOCATION", "STRING"),
        bigquery.SchemaField("INSURANCE", "STRING"),
        bigquery.SchemaField("LANGUAGE", "STRING"),
        bigquery.SchemaField("RELIGION", "STRING"),
        bigquery.SchemaField("MARITAL_STATUS", "STRING"),
        bigquery.SchemaField("ETHNICITY", "STRING"),
        bigquery.SchemaField("EDREGTIME", "TIMESTAMP"),
        bigquery.SchemaField("EDOUTTIME", "TIMESTAMP"),
        bigquery.SchemaField("DIAGNOSIS", "STRING"),
        bigquery.SchemaField("HOSPITAL_EXPIRE_FLAG", "INT64"),
        bigquery.SchemaField("HAS_CHARTEVENTS_DATA", "INT64"),
    ],
    "labevents": [
        bigquery.SchemaField("ROW_ID", "INT64"),
        bigquery.SchemaField("SUBJECT_ID", "INT64"),
        bigquery.SchemaField("HADM_ID", "INT64"),
        bigquery.SchemaField("ITEMID", "INT64"),
        bigquery.SchemaField("CHARTTIME", "TIMESTAMP"),
        bigquery.SchemaField("VALUE", "STRING"),
        bigquery.SchemaField("VALUENUM", "FLOAT64"),
        bigquery.SchemaField("VALUEUOM", "STRING"),
        bigquery.SchemaField("FLAG", "STRING"),
    ],
    "d_labitems": [
        bigquery.SchemaField("ROW_ID", "INT64"),
        bigquery.SchemaField("ITEMID", "INT64"),
        bigquery.SchemaField("LABEL", "STRING"),
        bigquery.SchemaField("FLUID", "STRING"),
        bigquery.SchemaField("CATEGORY", "STRING"),
        bigquery.SchemaField("LOINC_CODE", "STRING"),
    ],
    "outputevents": [
        bigquery.SchemaField("ROW_ID", "INT64"),
        bigquery.SchemaField("SUBJECT_ID", "INT64"),
        bigquery.SchemaField("HADM_ID", "INT64"),
        bigquery.SchemaField("ICUSTAY_ID", "INT64"),
        bigquery.SchemaField("CHARTTIME", "TIMESTAMP"),
        bigquery.SchemaField("ITEMID", "INT64"),
        bigquery.SchemaField("VALUE", "FLOAT64"),
        bigquery.SchemaField("VALUEUOM", "STRING"),
        bigquery.SchemaField("STORETIME", "TIMESTAMP"),
        bigquery.SchemaField("CGID", "INT64"),
        bigquery.SchemaField("STOPPED", "STRING"),
        bigquery.SchemaField("NEWBOTTLE", "INT64"),
        bigquery.SchemaField("ISERROR", "INT64"),
    ],
    "inputevents_cv": [
        bigquery.SchemaField("ROW_ID", "INT64"),
        bigquery.SchemaField("SUBJECT_ID", "INT64"),
        bigquery.SchemaField("HADM_ID", "INT64"),
        bigquery.SchemaField("ICUSTAY_ID", "INT64"),
        bigquery.SchemaField("CHARTTIME", "TIMESTAMP"),
        bigquery.SchemaField("ITEMID", "INT64"),
        bigquery.SchemaField("AMOUNT", "FLOAT64"),
        bigquery.SchemaField("AMOUNTUOM", "STRING"),
        bigquery.SchemaField("RATE", "FLOAT64"),
        bigquery.SchemaField("RATEUOM", "STRING"),
        bigquery.SchemaField("STORETIME", "TIMESTAMP"),
        bigquery.SchemaField("CGID", "INT64"),
        bigquery.SchemaField("ORDERID", "INT64"),
        bigquery.SchemaField("LINKORDERID", "INT64"),
        bigquery.SchemaField("STOPPED", "STRING"),
        bigquery.SchemaField("NEWBOTTLE", "INT64"),
        bigquery.SchemaField("ORIGINALAMOUNT", "FLOAT64"),
        bigquery.SchemaField("ORIGINALAMOUNTUOM", "STRING"),
        bigquery.SchemaField("ORIGINALROUTE", "STRING"),
        bigquery.SchemaField("ORIGINALRATE", "FLOAT64"),
        bigquery.SchemaField("ORIGINALRATEUOM", "STRING"),
        bigquery.SchemaField("ORIGINALSITE", "STRING"),
    ],
    "inputevents_mv": [
        bigquery.SchemaField("ROW_ID", "INT64"),
        bigquery.SchemaField("SUBJECT_ID", "INT64"),
        bigquery.SchemaField("HADM_ID", "INT64"),
        bigquery.SchemaField("ICUSTAY_ID", "INT64"),
        bigquery.SchemaField("STARTTIME", "TIMESTAMP"),
        bigquery.SchemaField("ENDTIME", "TIMESTAMP"),
        bigquery.SchemaField("ITEMID", "INT64"),
        bigquery.SchemaField("AMOUNT", "FLOAT64"),
        bigquery.SchemaField("AMOUNTUOM", "STRING"),
        bigquery.SchemaField("RATE", "FLOAT64"),
        bigquery.SchemaField("RATEUOM", "STRING"),
        bigquery.SchemaField("STORETIME", "TIMESTAMP"),
        bigquery.SchemaField("CGID", "INT64"),
        bigquery.SchemaField("ORDERID", "INT64"),
        bigquery.SchemaField("LINKORDERID", "INT64"),
        bigquery.SchemaField("ORDERCATEGORYNAME", "STRING"),
        bigquery.SchemaField("SECONDARYORDERCATEGORYNAME", "STRING"),
        bigquery.SchemaField("ORDERCOMPONENTTYPEDESCRIPTION", "STRING"),
        bigquery.SchemaField("ORDERCATEGORYDESCRIPTION", "STRING"),
        bigquery.SchemaField("PATIENTWEIGHT", "FLOAT64"),
        bigquery.SchemaField("TOTALAMOUNT", "FLOAT64"),
        bigquery.SchemaField("TOTALAMOUNTUOM", "STRING"),
        bigquery.SchemaField("ISOPENBAG", "INT64"),
        bigquery.SchemaField("CONTINUEINNEXTDEPT", "INT64"),
        bigquery.SchemaField("CANCELREASON", "INT64"),
        bigquery.SchemaField("STATUSDESCRIPTION", "STRING"),
        bigquery.SchemaField("COMMENTS_EDITEDBY", "STRING"),
        bigquery.SchemaField("COMMENTS_CANCELEDBY", "STRING"),
        bigquery.SchemaField("COMMENTS_DATE", "TIMESTAMP"),
        bigquery.SchemaField("ORIGINALAMOUNT", "FLOAT64"),
        bigquery.SchemaField("ORIGINALRATE", "FLOAT64"),
    ],
    "prescriptions": [
        bigquery.SchemaField("ROW_ID", "INT64"),
        bigquery.SchemaField("SUBJECT_ID", "INT64"),
        bigquery.SchemaField("HADM_ID", "INT64"),
        bigquery.SchemaField("ICUSTAY_ID", "INT64"),
        bigquery.SchemaField("STARTDATE", "TIMESTAMP"),
        bigquery.SchemaField("ENDDATE", "TIMESTAMP"),
        bigquery.SchemaField("DRUG_TYPE", "STRING"),
        bigquery.SchemaField("DRUG", "STRING"),
        bigquery.SchemaField("DRUG_NAME_POE", "STRING"),
        bigquery.SchemaField("DRUG_NAME_GENERIC", "STRING"),
        bigquery.SchemaField("FORMULARY_DRUG_CD", "STRING"),
        bigquery.SchemaField("GSN", "STRING"),
        bigquery.SchemaField("NDC", "STRING"),
        bigquery.SchemaField("PROD_STRENGTH", "STRING"),
        bigquery.SchemaField("DOSE_VAL_RX", "STRING"),
        bigquery.SchemaField("DOSE_UNIT_RX", "STRING"),
        bigquery.SchemaField("FORM_VAL_DISP", "STRING"),
        bigquery.SchemaField("FORM_UNIT_DISP", "STRING"),
        bigquery.SchemaField("ROUTE", "STRING"),
    ],
}


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def render_sql(template_path: Path, values: dict[str, Any]) -> str:
    sql = template_path.read_text(encoding="utf-8")
    for key, value in values.items():
        sql = sql.replace("{{" + key + "}}", str(value))
    unresolved = [part.split("}}", 1)[0] for part in sql.split("{{")[1:]]
    if unresolved:
        raise ValueError(f"Unresolved SQL template variables: {sorted(set(unresolved))}")
    return sql


def dataset_ref(config: dict[str, Any]) -> bigquery.DatasetReference:
    return bigquery.DatasetReference(config["project_id"], config["dataset_id"])


def ensure_dataset(client: bigquery.Client, config: dict[str, Any]) -> None:
    dataset = bigquery.Dataset(dataset_ref(config))
    dataset.location = config.get("location", "EU")
    try:
        client.create_dataset(dataset, exists_ok=True)
        return
    except exceptions.Forbidden:
        dataset_id = f"{config['project_id']}.{config['dataset_id']}"
        try:
            client.get_dataset(dataset_id)
            print(f"Dataset {dataset_id} already exists; continuing without create permission.")
            return
        except exceptions.Forbidden as exc:
            raise PermissionError(
                "BigQuery permission error. Your account cannot create or access "
                f"dataset {dataset_id}. Ask the project owner for BigQuery Admin, "
                "or at least BigQuery Data Editor + BigQuery Job User on this project."
            ) from exc
        except exceptions.NotFound as exc:
            raise PermissionError(
                "BigQuery permission error. The dataset does not exist and your "
                f"account cannot create it: {dataset_id}. Ask the project owner to "
                "create the dataset or grant BigQuery Admin."
            ) from exc


def raw_uri(config: dict[str, Any], table_key: str) -> str:
    gcs = config["gcs"]
    file_name = config["raw_files"][table_key]
    return f"gs://{gcs['bucket']}/{gcs['raw_prefix'].strip('/')}/{file_name}"


def load_raw_tables(client: bigquery.Client, config: dict[str, Any], table_keys: list[str] | None = None) -> None:
    ensure_dataset(client, config)

    tables_to_load = config["tables"]
    if table_keys:
        unknown = sorted(set(table_keys) - set(tables_to_load))
        if unknown:
            raise ValueError(f"Unknown table keys for --load-only: {unknown}")
        tables_to_load = {key: tables_to_load[key] for key in table_keys}

    for table_key, table_name in tables_to_load.items():
        if table_key not in TABLE_SCHEMAS:
            raise ValueError(f"No BigQuery schema is defined for table key: {table_key}")

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            schema=TABLE_SCHEMAS[table_key],
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            allow_quoted_newlines=True,
        )
        table_id = f"{config['project_id']}.{config['dataset_id']}.{table_name}"
        uri = raw_uri(config, table_key)
        print(f"Loading {uri} -> {table_id}")
        job = client.load_table_from_uri(uri, table_id, job_config=job_config)
        job.result()
        table = client.get_table(table_id)
        print(f"Loaded {table.num_rows} rows into {table_id}")


def build_template_values(config: dict[str, Any]) -> dict[str, Any]:
    tables = config["tables"]
    outputs = config["outputs"]
    table_values = {f"{key}_table": value for key, value in tables.items()}
    output_values = {f"{key}": value for key, value in outputs.items()}
    return {
        "project_id": config["project_id"],
        "dataset_id": config["dataset_id"],
        "window_hours": config["window_hours"],
        "top_n_items": config["top_n_items"],
        "top_n_chart_items": config.get("top_n_chart_items", config["top_n_items"]),
        "top_n_lab_items": config.get("top_n_lab_items", 30),
        "top_n_output_items": config.get("top_n_output_items", 20),
        "top_n_input_items": config.get("top_n_input_items", 30),
        "top_n_drugs": config.get("top_n_drugs", 30),
        **table_values,
        **output_values,
    }


def run_preprocessing_query(
    client: bigquery.Client,
    config: dict[str, Any],
    dry_run: bool,
    sql_template: Path,
) -> None:
    sql = render_sql(sql_template, build_template_values(config))
    job_config = bigquery.QueryJobConfig(
        use_legacy_sql=False,
        dry_run=dry_run,
        use_query_cache=False,
    )
    print("Running BigQuery preprocessing SQL")
    job = client.query(sql, job_config=job_config, location=config.get("location"))
    if dry_run:
        print(f"Dry run bytes processed: {job.total_bytes_processed:,}")
        return
    job.result()
    print("BigQuery preprocessing complete")


def export_features(client: bigquery.Client, config: dict[str, Any]) -> None:
    uri = config["outputs"]["features_export_uri"]
    table_id = f"{config['project_id']}.{config['dataset_id']}.{config['outputs']['features_table']}"
    job_config = bigquery.ExtractJobConfig(
        destination_format=bigquery.DestinationFormat.CSV,
        print_header=True,
    )
    print(f"Exporting {table_id} -> {uri}")
    job = client.extract_table(table_id, uri, job_config=job_config, location=config.get("location"))
    job.result()
    print("Export complete")


def download_exports(config: dict[str, Any], output_dir: Path) -> None:
    uri = config["outputs"]["features_export_uri"]
    if not uri.startswith("gs://"):
        raise ValueError("features_export_uri must start with gs://")

    without_scheme = uri.removeprefix("gs://")
    bucket_name, blob_pattern = without_scheme.split("/", 1)
    prefix = blob_pattern.split("*", 1)[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client(project=config["project_id"])
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise ValueError(f"No exported files found at {uri}")

    for blob in blobs:
        destination = output_dir / Path(blob.name).name
        print(f"Downloading gs://{bucket_name}/{blob.name} -> {destination}")
        blob.download_to_filename(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BigQuery ICU LOS preprocessing pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/bigquery.yaml"))
    parser.add_argument("--load-raw", action="store_true", help="Load raw CSV files from GCS into BigQuery first.")
    parser.add_argument(
        "--load-only",
        nargs="+",
        help="Load only selected table keys from the config, for example: --load-only outputevents prescriptions.",
    )
    parser.add_argument("--skip-query", action="store_true", help="Skip preprocessing SQL execution.")
    parser.add_argument("--export", action="store_true", help="Export the BigQuery features table to GCS.")
    parser.add_argument("--download", type=Path, help="Download exported feature shards to this local directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate query and estimate bytes without running it.")
    parser.add_argument("--sql-template", type=Path, default=SQL_TEMPLATE, help="SQL template to render and run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    client = bigquery.Client(project=config["project_id"])

    if args.load_raw or args.load_only:
        load_raw_tables(client, config, args.load_only)
    else:
        ensure_dataset(client, config)

    if not args.skip_query:
        run_preprocessing_query(client, config, args.dry_run, args.sql_template)

    if args.export and not args.dry_run:
        export_features(client, config)

    if args.download and not args.dry_run:
        download_exports(config, args.download)


if __name__ == "__main__":
    main()
