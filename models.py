from pathlib import Path
from typing import NamedTuple

import attrs
from streamlit.runtime.uploaded_file_manager import UploadedFile

from enums import TaskType


class Upload_PDF(NamedTuple):
    id: str
    file: UploadedFile


class Downloaded_PDF(NamedTuple):
    id: str
    file_path: Path


class Load_XML(NamedTuple):
    id: str
    path: Path | None


@attrs.define(frozen=True)
class Paper:
    id: str = attrs.field(repr=False)
    title: str
    pdf_path: Path = attrs.field(repr=False)
    xml_path: Path = attrs.field(repr=False)


@attrs.define(on_setattr=attrs.setters.frozen)
class Task:
    id: str = attrs.field(repr=False)
    paper: Paper
    type: TaskType
    verify: bool
    text: str = attrs.field(repr=False)
    parent_task_id: str | None = attrs.field(repr=True, default=None)
    time_elapsed: float = attrs.field(on_setattr=attrs.setters.NO_OP, default=None)
    extracted_ents: set[str] = attrs.field(on_setattr=attrs.setters.NO_OP, factory=set)
    pending: bool = attrs.field(on_setattr=attrs.setters.NO_OP, default=True)


class TasksBatchDone(NamedTuple):
    batch_id: str
    exec_time: float
    results: list[Task]


@attrs.define(frozen=True)
class Color:
    r: float
    g: float
    b: float

    def invert(self):
        return Color(1 - self.r, 1 - self.g, 0)

    def get(self):
        return self.r, self.g, self.b
