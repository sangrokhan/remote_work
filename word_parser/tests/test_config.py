import textwrap
import pytest
from core.config import Config, load_config


SAMPLE_YAML = textwrap.dedent("""\
    heading_styles:
      "Heading 1": 1
      "Heading 2": 2
    font_size_map:
      24: 1
      18: 2
    heading_tags:
      "3.2 Configuration": "config"
    table_merge:
      enabled: true
    output_dir: "output"
    log_level: "INFO"
""")


def test_load_config_parses_heading_styles(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(SAMPLE_YAML)
    cfg = load_config(str(cfg_file))
    assert cfg.heading_styles["Heading 1"] == 1
    assert cfg.heading_styles["Heading 2"] == 2


def test_load_config_parses_font_size_map(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(SAMPLE_YAML)
    cfg = load_config(str(cfg_file))
    assert cfg.font_size_map[24] == 1
    assert cfg.font_size_map[18] == 2


def test_load_config_parses_heading_tags(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(SAMPLE_YAML)
    cfg = load_config(str(cfg_file))
    assert cfg.heading_tags["3.2 Configuration"] == "config"


def test_load_config_table_merge(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(SAMPLE_YAML)
    cfg = load_config(str(cfg_file))
    assert cfg.table_merge_enabled is True


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")
