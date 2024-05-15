from pathlib import Path
from typing import NamedTuple
import attrs
from streamlit.runtime.uploaded_file_manager import UploadedFile


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
    id: str
    xml_path: Path
    title: str
