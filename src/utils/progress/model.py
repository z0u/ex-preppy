from dataclasses import dataclass


@dataclass(slots=True)
class BarData:
    count: int
    total: int
    description: str
    suffix: str
    elapsed_time: float

    @property
    def fraction(self) -> float:
        return (self.count / self.total) if self.total > 0 else 1
