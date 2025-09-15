from enum import Enum


class QFormat(str, Enum):
    MC = "MC"
    BINARY = "binary"
    OPEN = "open-ended"


class Template(str, Enum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"


class QuestionTone(str, Enum):
    ORIGINAL = "original"
    NEUTRAL = "neutral"
    WORRIED = "worried"
