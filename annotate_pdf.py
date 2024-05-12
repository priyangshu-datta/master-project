# import fitz
import pymupdf as pypdf
import random
from pathlib import Path
from icecream import ic


def annotate_pdf(pdf_path, entity_type, entities):
    doc = pypdf.open(pdf_path)
    stroke_color = {
        entity: (random.random(), random.random(), 0) # research papers are filled with blues already
        for entity in entities
    }
    for pi in range(doc.page_count):
        page = doc[pi]
        entities_instances = {}
        for entity in entities:
            entities_instances[entity] = page.search_for(f" {entity} ")

        for entity, instances in entities_instances.items():
            for inst in instances:
                annot = page.add_highlight_annot(inst)
                annot.set_colors(stroke=stroke_color[entity], fill=stroke_color[entity])
                annot.set_opacity(0.5)
                annot.update()

    doc.save(Path(f"temp/views/{entity_type}/").joinpath(pdf_path.name + ''.join(pdf_path.suffixes)))
    doc.close()
