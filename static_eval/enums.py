from enum import Enum


class QFormat(str, Enum):
    MC = "MC"
    BINARY = "binary"
    OPEN = "open-ended"


class Perturbation(str, Enum):
    SYCOPHANCY = "sycophancy"
    FORMAT_MC = "format_mc"
    FORMAT_BINARY = "format_binary"
