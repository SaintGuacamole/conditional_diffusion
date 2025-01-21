from enum import Enum


class Normalization(str, Enum):
    NONE = 0
    ZERO_ONE_TO_MINUS_ONE_ONE = 1
    MINUS_ONE_ONE_TO_ZERO_ONE = 2