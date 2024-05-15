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
    time_elapsed: float = attrs.field(on_setattr=attrs.setters.NO_OP, default=None)
    extracted_ents: set[str] = attrs.field(on_setattr=attrs.setters.NO_OP, factory=set)
    pending: bool = attrs.field(on_setattr=attrs.setters.NO_OP, default=True)
