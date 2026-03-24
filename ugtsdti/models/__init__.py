# ugtsdti/models/__init__.py

from .fusion.ug import UncertaintyGatedFusion
from .hybrid import HybridDTIModel
from .student.baseline import BaselineStudent
from .student.plm import ESMProteinStudent
from .teacher.baseline import BaselineTeacher
from .teacher.gcn_teacher import GCNTeacher

__all__ = [
    "BaselineStudent",
    "BaselineTeacher",
    "GCNTeacher",
    "UncertaintyGatedFusion",
    "HybridDTIModel",
    "ESMProteinStudent",
]
