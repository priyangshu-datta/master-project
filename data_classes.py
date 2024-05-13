from pathlib import Path
import attrs
import numpy as np
import torch


@attrs.define(on_setattr=attrs.setters.frozen)
class TaskObject:
    paper_id: str
    entity_type: str
    verify: bool
    xml_path: Path
    chunks: list[str]
    query_embeds: list[torch.Tensor] | np.ndarray | torch.Tensor
    temperature: float
    extracted_ents: None | list[str] = attrs.field(
        on_setattr=attrs.setters.NO_OP, factory=list
    )


@attrs.define(frozen=True)
class Paper:
    path: Path
    title: str
    paper_id: str
    