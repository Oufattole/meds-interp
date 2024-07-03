import subprocess
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

from meds_interp.long_df import generate_long_df


def run_command(script: str, args: list[str], hydra_kwargs: dict[str, str], test_name: str):
    command_parts = [script] + args + [f"'{k}={v}'" for k, v in hydra_kwargs.items()]
    command_str = " ".join(command_parts)
    logger.info(command_str)
    command_out = subprocess.run(command_str, shell=True, capture_output=True)
    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    if command_out.returncode != 0:
        raise AssertionError(f"{test_name} failed!\nstdout:\n{stdout}\nstderr:\n{stderr}")
    return stderr, stdout


def test_knn_tuning(tmp_path):
    train_df = pl.DataFrame(
        {
            "modality_1": [[1, 1, 0], [1, 2, 1], [2, 0, -2], [0, 1, 2]],
            "modality_2": [[1, 0, 0], [0, 2, 1], [4, 1, 1], [3, 1, 2]],
            "label": [0, 0, 1, 1],
        }
    )
    val_df = pl.DataFrame(
        {
            "modality_1": [[3, 5, 2], [4, 1, 3], [6, 2, 0], [7, 1, 1]],
            "modality_2": [[2, 3, 1], [5, 0, 4], [1, 6, 2], [3, 2, 5]],
            "label": [0, 1, 0, 1],
        }
    )
    test_df = pl.DataFrame(
        {
            "modality_1": [[-2, 3, -1], [0, -1, 2], [-3, 1, -2], [1, -2, 3]],
            "modality_2": [[3, -1, 0], [-2, 2, -3], [1, -3, 2], [0, 3, -1]],
            "label": [0, 0, 1, 1],
        }
    )
    train_df.write_parquet(Path(tmp_path) / "train.parquet")
    val_df.write_parquet(Path(tmp_path) / "val.parquet")
    test_df.write_parquet(Path(tmp_path) / "test.parquet")

    test_config = dict(
        # modalities='["modality_1","modality_2"]',
        # modality_weights='[1,1]',
        input_path=tmp_path,
    )

    stderr, stdout = run_command(
        "meds-knn",
        ["--multirun", "$(generate-weights [modality_1,modality_2])"],
        test_config,
        "test knn",
    )


def test_MEDS_Tab(tmp_path):
    train_df = pl.DataFrame(
        {
            "patient_id": [1, 1, 2],
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 3, 1)],
            "label": [0, 1, 1],
            "Embedding_1": [[0.1, 0.2], [0.3, 0.4], [0.2, 0.4]],
            "Embedding_2": [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
        }
    )
    val_df = pl.DataFrame(
        {
            "patient_id": [1, 1, 2],
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 3, 1)],
            "label": [0, 1, 1],
            "Embedding_1": [[0.1, 0.2], [0.3, 0.4], [0.2, 0.4]],
            "Embedding_2": [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
        }
    )
    test_df = pl.DataFrame(
        {
            "patient_id": [1, 1, 2],
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 3, 1)],
            "label": [0, 1, 1],
            "Embedding_1": [[0.1, 0.2], [0.3, 0.4], [0.2, 0.4]],
            "Embedding_2": [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
        }
    )

    train_output_path = Path(tmp_path) / "final_cohort/train/0.parquet"
    train_output_path.parent.mkdir(parents=True)

    val_output_path = Path(tmp_path) / "final_cohort/val/0.parquet"
    val_output_path.parent.mkdir(parents=True)

    test_output_path = Path(tmp_path) / "final_cohort/test/0.parquet"
    test_output_path.parent.mkdir(parents=True)

    generate_long_df(train_df).write_parquet(train_output_path)
    generate_long_df(val_df).write_parquet(val_output_path)
    generate_long_df(test_df).write_parquet(test_output_path)

    meds_tab_config = dict(MEDS_cohort_dir=tmp_path)
    run_command("meds-tab-describe", [], meds_tab_config, "describe_codes")

    meds_tab_tabularize = dict(
        MEDS_cohort_dir=tmp_path,
        do_overwrite=False,
    )
    meds_tab_tabularize["tabularization.min_code_inclusion_frequency"] = 10
    meds_tab_tabularize["tabularization.window_sizes"] = "[1d,30d,365d,full]"
    meds_tab_tabularize["tabularization.min_code_inclusion_frequency"] = 0
    meds_tab_tabularize["tabularization.aggs"] = "[value/count,value/sum,value/sum_sqd,value/min,value/max]"
    run_command("meds-tab-tabularize-static", [], meds_tab_tabularize, "tabularize time series data")
    args = ["--multirun", "hydra/launcher=joblib", "'worker=range(0,1)'"]
    run_command("meds-tab-tabularize-time-series", args, meds_tab_tabularize, "tabularize time series data")
