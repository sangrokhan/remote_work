import re
from dataclasses import dataclass, field


@dataclass
class ParsedFeature:
    feature_id: str
    name: str
    gen: str
    category: str
    description: str
    kpi_impacts: list[dict] = field(default_factory=list)
    controlling_params: list[dict] = field(default_factory=list)
    dependencies: list[dict] = field(default_factory=list)


def _parse_table(lines: list[str]) -> list[dict]:
    """Parse a markdown table (with header row) into list of dicts."""
    rows = [l for l in lines if l.startswith("|") and not re.match(r"\|[-| ]+\|", l)]
    if len(rows) < 2:
        return []
    headers = [h.strip() for h in rows[0].strip("|").split("|")]
    result = []
    for row in rows[1:]:
        values = [v.strip() for v in row.strip("|").split("|")]
        result.append(dict(zip(headers, values)))
    return result


def _extract_section(text: str, section_name: str) -> list[str]:
    """Return lines belonging to the named ## section."""
    lines = text.splitlines()
    in_section = False
    result = []
    for line in lines:
        if line.startswith("## ") and section_name.lower() in line.lower():
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section:
            result.append(line)
    return result


def parse_feature_doc(text: str) -> ParsedFeature:
    feature_id = re.search(r"\*\*Feature ID\*\*:\s*(\S+)", text)
    gen = re.search(r"\*\*Generation\*\*:\s*(\S+)", text)
    category = re.search(r"\*\*Category\*\*:\s*(\S+)", text)
    name_match = re.match(r"#\s+Feature:\s+(.+)", text)

    desc_lines = _extract_section(text, "Description")
    description = " ".join(l.strip() for l in desc_lines if l.strip())

    kpi_lines = _extract_section(text, "KPI Impact")
    kpi_rows = _parse_table(kpi_lines)
    kpi_impacts = []
    for row in kpi_rows:
        kpi_impacts.append({
            "kpi_id": row.get("KPI ID", "").strip(),
            "kpi_name": row.get("KPI Name", "").strip(),
            "direction": row.get("Direction", "").strip(),
            "magnitude": row.get("Magnitude", "").strip(),
            "condition": row.get("Condition", "").strip(),
        })

    param_lines = _extract_section(text, "Controlling Parameters")
    param_rows = _parse_table(param_lines)
    controlling_params = [
        {
            "param_id": r.get("Parameter ID", "").strip(),
            "effect": r.get("Effect When Increased", "").strip(),
        }
        for r in param_rows
    ]

    dep_lines = _extract_section(text, "Feature Dependencies")
    dep_rows = _parse_table(dep_lines)
    dependencies = [
        {
            "feature_id": r.get("Feature ID", "").strip(),
            "dep_type": r.get("Dependency Type", "").strip(),
        }
        for r in dep_rows
    ]

    return ParsedFeature(
        feature_id=feature_id.group(1) if feature_id else "",
        name=name_match.group(1).strip() if name_match else "",
        gen=gen.group(1) if gen else "",
        category=category.group(1) if category else "",
        description=description,
        kpi_impacts=kpi_impacts,
        controlling_params=controlling_params,
        dependencies=dependencies,
    )
