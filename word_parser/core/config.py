from dataclasses import dataclass
import yaml


@dataclass
class Config:
    heading_styles: dict[str, int]
    font_size_map: dict[int, int]
    heading_tags: dict[str, str]
    table_merge_enabled: bool
    output_dir: str
    log_level: str
    chunk_split_depth: int = 0  # 0 = split on all headings (flat); N >= 1 = folder structure


def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    font_size_map = {int(k): int(v) for k, v in (data.get("font_size_map") or {}).items()}

    return Config(
        heading_styles=data.get("heading_styles") or {},
        font_size_map=font_size_map,
        heading_tags=data.get("heading_tags") or {},
        table_merge_enabled=(data.get("table_merge") or {}).get("enabled", True),
        output_dir=data.get("output_dir", "output"),
        log_level=data.get("log_level", "INFO"),
        chunk_split_depth=int(data.get("chunk_split_depth", 0)),
    )
