import csv
import io
from pathlib import Path
from typing import Union, Tuple, List

from causality_graph.schema import ParameterNode


def _parse_float(val: str):
    val = val.strip()
    if val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def parse_param_csv(
    source: Union[Path, io.StringIO],
    return_metadata: bool = False,
) -> Union[List[ParameterNode], Tuple[List[ParameterNode], List[dict]]]:
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(text))
    else:
        source.seek(0)
        reader = csv.DictReader(source)

    nodes = []
    metadata = []
    for row in reader:
        node = ParameterNode(
            id=row["id"].strip(),
            name=row["name"].strip(),
            data_type=row["data_type"].strip(),
            range_min=_parse_float(row.get("range_min", "")),
            range_max=_parse_float(row.get("range_max", "")),
            default_value=row.get("default_value", "").strip(),
            unit=row.get("unit", "").strip(),
            description=row.get("description", "").strip(),
        )
        nodes.append(node)
        metadata.append({
            "id": node.id,
            "linked_feature_id": row.get("linked_feature_id", "").strip(),
        })

    if return_metadata:
        return nodes, metadata
    return nodes
