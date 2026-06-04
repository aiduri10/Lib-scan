from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ScanResult:
    data: str
    barcode_type: str
    rect: Tuple[int, int, int, int]
    physical_pattern: Optional[str] = None
