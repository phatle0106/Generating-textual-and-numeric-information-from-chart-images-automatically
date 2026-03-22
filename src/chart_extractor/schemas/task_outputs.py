from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Polygon:
    x0: int
    y0: int
    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int


@dataclass
class TextBlock:
    id: int
    polygon: Polygon
    text: str
    score: float

