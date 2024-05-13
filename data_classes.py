from pathlib import Path
import attrs
import numpy as np
import torch

from enums import EntityType


@attrs.define(on_setattr=attrs.setters.frozen)
class TaskObject:
    task_id: str
    paper_id: str
    entity_type: EntityType
    verify: bool
    xml_path: Path = attrs.field(repr=False)
    chunks: list[str] = attrs.field(repr=False)
    query_embeds: list[torch.Tensor] | np.ndarray | torch.Tensor = attrs.field(
        repr=False
    )
    temperature: float = attrs.field(repr=False)
    extracted_ents: set[str] = attrs.field(on_setattr=attrs.setters.NO_OP, factory=set)
    pending: bool = attrs.field(on_setattr=attrs.setters.NO_OP, default=True)
    time_elapsed: float = attrs.field(on_setattr=attrs.setters.NO_OP, default=0.0)


@attrs.define(frozen=True)
class Paper:
    path: Path
    title: str
    paper_id: str
