# import fitz
import pymupdf as pypdf
import random
from pathlib import Path
from icecream import ic
from enums import TaskType
import typing as t
from models import Color


def annotate_pdf(pdf_path: Path, entities: t.Dict[TaskType, t.Set[str]]):
    doc: pypdf.Document = pypdf.open(pdf_path)

    colors = {
        ent_type: {
            ent: Color(
                random.random(), random.random(), 0
            )  # research papers are filled with blues already
            for ent in ents
        }
        for ent_type, ents in entities.items()
    }

    for pi in range(doc.page_count):
        page = doc.load_page(pi)
        entities_instances: t.Dict[TaskType, t.Dict[str, t.List[pypdf.Rect]]] = {}

        for ent_type, ents in entities.items():
            entities_instances[ent_type] = {}
            for ent in ents:
                if ent in entities_instances[ent_type]:
                    entities_instances[ent_type][ent].extend(
                        pypdf.utils.search_for(page, f" {ent} ")
                    )
                else:
                    entities_instances[ent_type][ent] = pypdf.utils.search_for(
                        page, f" {ent} "
                    )

        for ent_type, ents_rects in entities_instances.items():
            for ent, rects in ents_rects.items():
                for rect in rects:
                    annot = page.add_rect_annot(rect)
                    pypdf.utils.draw_rect(
                        page,
                        pypdf.Rect(
                            x0=rect.tl[0],
                            y0=rect.tl[1] - 9.5,
                            x1=rect.tl[0] + 8,
                            y1=rect.tl[1] + 1,
                        ),
                        fill=colors[ent_type][ent].get(),
                        stroke_opacity=0,
                    )
                    pypdf.utils.insert_textbox(
                        page,
                        pypdf.Rect(
                            x0=rect.tl[0],
                            y0=rect.tl[1] - 12,
                            x1=rect.tl[0] + 10,
                            y1=rect.tl[1] + 7,
                        ),
                        buffer=str(ent_type).upper()[0],
                        fontsize=10,
                        color=(1, 1, 1),
                    )

                    annot.set_colors(
                        stroke=colors[ent_type][ent].get(),
                        fill=colors[ent_type][ent].get(),
                    )
                    annot.set_opacity(0.5)
                    annot.update()

    # doc.save(Path(f"temp/views/").joinpath(pdf_path.name))
    annot_pdf = doc.tobytes()

    doc.close()

    return annot_pdf
