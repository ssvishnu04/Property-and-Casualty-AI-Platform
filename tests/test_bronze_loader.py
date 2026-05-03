from pathlib import Path

from src.data_processing.bronze_loader import BRONZE_OUTPUTS


def test_bronze_output_paths_defined():
    assert "claims" in BRONZE_OUTPUTS
    assert "fnol" in BRONZE_OUTPUTS
    assert "ingestion_audit" in BRONZE_OUTPUTS


def test_bronze_root_folder_can_exist():
    bronze_root = Path("data/bronze")
    bronze_root.mkdir(parents=True, exist_ok=True)

    assert bronze_root.exists()