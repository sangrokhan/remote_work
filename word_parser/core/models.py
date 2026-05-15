from dataclasses import dataclass, field


@dataclass
class Run:
    text: str
    font_size: float | None
    bold: bool


@dataclass
class ParagraphElement:
    text: str
    style_name: str
    runs: list[Run]
    page_approx: int
    is_page_break: bool = False


@dataclass
class TableElement:
    rows: list[list[str]]
    col_count: int
    page_approx: int
    preceded_by_page_break: bool


@dataclass
class ImageElement:
    relationship_id: str
    content_type: str
    data: bytes
    page_approx: int


@dataclass
class Chunk:
    heading_text: str
    heading_depth: int
    tag: str
    elements: list
    index: int
    folder_slugs: list = field(default_factory=list)
    table_counter: int = 0
    image_counter: int = 0
