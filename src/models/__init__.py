"""
Model implementations for embedding distillation
"""

from .embedding_model import EmbeddingModel
from .teacher_model import TeacherModel
from .student_model import StudentModel

__all__ = ["EmbeddingModel", "TeacherModel", "StudentModel"]