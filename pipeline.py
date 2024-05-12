import pydash as py_
from processing import sentence_splitter, group_sentences, sub_ci
from enums import EntityType
from utils import query_embedder, load_pdfs, pdfs_to_xmls, xml_to_body_text, extract_entities
from icecream import ic

chunker = py_.flow(
    sentence_splitter,
    lambda x: group_sentences(x, 200, 2),
)

clean_entities = (
    lambda entities: py_.chain(entities)
    .map_(sub_ci(" +", " "))
    .filter_(lambda x: len(x) > 0)
    .apply(list)
    .value()
)

pdfs_to_text = lambda urls: py_.chain(urls).apply(load_pdfs).apply(
    lambda x: pdfs_to_xmls(x[0], x[1])
).apply(py_.to_pairs).map_(
    lambda id_path: (id_path[0], xml_to_body_text(id_path[1]))
)

dmddID_to_text = lambda ids, dmdd: py_.chain(ids).reduce_(
    lambda acc, id: py_.set_(acc, id, dmdd['text'][f'{id}']), {}
).apply(py_.to_pairs)


text_to_entities = lambda text_chain, entity_type, verify=True, temperature=None: text_chain.map_(
    lambda id_text: (id_text[0], chunker(id_text[1]))
).tap(lambda _: ic("Text chunked")).map_(
    lambda id_chunks: (
        id_chunks[0],
        extract_entities(id_chunks[1], query_embedder(entity_type), entity_type, verify=verify, temperature=temperature),
    )
).tap(lambda _: ic("extracted and verified")).map_(
    lambda id_entities: (id_entities[0], clean_entities(id_entities[1]))
).apply(
    py_.from_pairs
).value()