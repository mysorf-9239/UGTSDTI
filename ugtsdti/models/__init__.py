# ugtsdti/models/__init__.py
from .baseline import BaselineModel
from .fusion.pairgate import PairGateFusion
from .hybrid import HybridDTIModel
from .student.cnn1d import CNN1DStudent
from .student.plm import ESMProteinStudent

__all__ = [
    "BaselineModel",
    "PairGateFusion",
    "HybridDTIModel",
    "CNN1DStudent",
    "ESMProteinStudent",
]
