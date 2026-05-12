from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    CONTROL_GROUP = "control_group"
    FUNCTION = "function"
    FEATURE = "feature"
    LAYER = "layer"
    KPI = "kpi"
    PARAMETER = "parameter"


class EdgeType(str, Enum):
    INCLUDES = "INCLUDES"                  # ControlGroup → Function
    REALIZED_BY = "REALIZED_BY"            # Function → Feature
    MEASURED_BY = "MEASURED_BY"            # Function/Feature → KPI
    IMPLEMENTED_IN = "IMPLEMENTED_IN"      # Feature → Layer
    TUNED_BY = "TUNED_BY"                  # Feature → Parameter
    DEPENDS_ON = "DEPENDS_ON"              # Feature → Feature
    AFFECTS = "AFFECTS"                    # Parameter → KPI
    CONFLICTS_WITH = "CONFLICTS_WITH"      # Parameter → Parameter


class Direction(str, Enum):
    POSITIVE = "+"
    NEGATIVE = "-"


class Magnitude(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Layer(str, Enum):
    PHY = "PHY"
    MAC = "MAC"
    RLC = "RLC"
    PDCP = "PDCP"
    SDAP = "SDAP"
    RRC = "RRC"
    NAS = "NAS"


@dataclass
class ControlGroupNode:
    id: str
    name: str
    description: str = ""


@dataclass
class FunctionNode:
    id: str
    name: str
    description: str = ""


@dataclass
class FeatureNode:
    id: str
    name: str
    layer: Layer
    description: str = ""


@dataclass
class LayerNode:
    id: str
    name: Layer


@dataclass
class KPINode:
    id: str
    name: str
    unit: str
    good_direction: Direction
    category: str = ""
    layer: str = ""
    spec_ref: str = ""
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
    spec_ref: str = ""
    description: str = ""


@dataclass
class Edge:
    from_id: str
    to_id: str
    relation: EdgeType
    direction: Optional[Direction] = None
    magnitude: Optional[Magnitude] = None
    confidence: float = 1.0
    validated: bool = False
    notes: str = ""
