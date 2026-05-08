from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    KPI = "kpi"
    FEATURE = "feature"
    PARAMETER = "parameter"


class EdgeType(str, Enum):
    AFFECTS = "AFFECTS"
    CONTROLLED_BY = "CONTROLLED_BY"
    DEPENDS_ON = "DEPENDS_ON"
    CORRELATES = "CORRELATES"


class Direction(str, Enum):
    POSITIVE = "+"
    NEGATIVE = "-"


class Generation(str, Enum):
    G4 = "4G"
    G5 = "5G"
    BOTH = "both"


class Magnitude(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class KPINode:
    id: str
    name: str
    unit: str
    good_direction: Direction
    description: str = ""


@dataclass
class FeatureNode:
    id: str
    name: str
    gen: Generation
    category: str
    description: str = ""


@dataclass
class ParameterNode:
    id: str
    name: str
    data_type: str
    range_min: Optional[float]
    range_max: Optional[float]
    default_value: str
    unit: str = ""
    description: str = ""


@dataclass
class Edge:
    from_id: str
    to_id: str
    relation: EdgeType
    direction: Optional[Direction] = None
    magnitude: Optional[Magnitude] = None
    condition: str = ""
    confidence: float = 1.0
    validated: bool = False
    notes: str = ""
