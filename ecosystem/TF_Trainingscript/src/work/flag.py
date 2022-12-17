from enum import Enum

class Type(Enum):
    ABSENCE = 0
    EXIST = 1
    LINENUM = 2
    BLOCK = 3
    LOG = '[\[]EVENT[\]]'