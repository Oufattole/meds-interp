import subprocess
from pathlib import Path

import polars as pl
from loguru import logger


def run_command(script: str, args: list[str], hydra_kwargs: dict[str, str], test_name: str):
    command_parts = [script] + args + [f"{k}={v}" for k, v in hydra_kwargs.items()]
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
