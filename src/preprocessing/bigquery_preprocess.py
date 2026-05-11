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


def load_raw_tables(client: bigquery.Client, config: dict[str, Any]) -> None:
    ensure_dataset(client, config)
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    for table_key, table_name in config["tables"].items():
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
    return {
        "project_id": config["project_id"],
        "dataset_id": config["dataset_id"],
        "window_hours": config["window_hours"],
        "top_n_items": config["top_n_items"],
        "chartevents_table": tables["chartevents"],
        "icustays_table": tables["icustays"],
        "d_items_table": tables["d_items"],
        "patients_table": tables["patients"],
        "admissions_table": tables["admissions"],
        "selected_items_table": outputs["selected_items_table"],
        "features_table": outputs["features_table"],
        "quality_report_table": outputs["quality_report_table"],
    }


def run_preprocessing_query(client: bigquery.Client, config: dict[str, Any], dry_run: bool) -> None:
    sql = render_sql(SQL_TEMPLATE, build_template_values(config))
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
    parser.add_argument("--skip-query", action="store_true", help="Skip preprocessing SQL execution.")
    parser.add_argument("--export", action="store_true", help="Export the BigQuery features table to GCS.")
    parser.add_argument("--download", type=Path, help="Download exported feature shards to this local directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate query and estimate bytes without running it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    client = bigquery.Client(project=config["project_id"])

    if args.load_raw:
        load_raw_tables(client, config)
    else:
        ensure_dataset(client, config)

    if not args.skip_query:
        run_preprocessing_query(client, config, args.dry_run)

    if args.export and not args.dry_run:
        export_features(client, config)

    if args.download and not args.dry_run:
        download_exports(config, args.download)


if __name__ == "__main__":
    main()
