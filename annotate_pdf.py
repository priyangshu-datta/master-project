# import fitz
import pymupdf as pypdf
import random
from pathlib import Path
from icecream import ic


def annotate_pdf(pdf_path, entities):
    doc = pypdf.open(pdf_path)
    stroke_color = {
        entity: (random.random(), random.random(), random.random())
        for entity in entities
    }
    for pi in range(doc.page_count):
        page = doc[pi]
        entities_instances = {}
        for entity in entities:
            entities_instances[entity] = page.search_for(entity)

        five_percent_height = (page.rect.br.y - page.rect.tl.y) * 0.05
        
        for entity, instances in entities_instances.items():
            for inst in instances:
                annot = page.add_rect_annot(inst)
                annot.set_colors(stroke=stroke_color[entity])
                annot.update()

    doc.save(Path("temp/views").joinpath(pdf_path.name))
    doc.close()
