from dotenv import load_dotenv

load_dotenv()

import hashlib as hashlib
import re as re
import random
import time as time
import typing
from enum import StrEnum, auto
from pathlib import Path as Path

import pydash
from loguru import logger

lg = logger
py_ = pydash
t = typing
rand = random
LLM_TEMPERATURE = 0.06
SENTENCES_PER_PAPER = 150


class TaskType(StrEnum):
    DATASET = auto()
    BASELINE = auto()


class DatasetSearchEngine(StrEnum):
    KAGGLE = auto()
    HF = auto()
    PwC = auto()
