from pathlib import Path
import shutil
import time
import pydash as py_
import os
from bs4 import BeautifulSoup as bs4
import re
from urllib import request, parse
import google.ai.generativelanguage as glm
from processing import embedder, sub_ci
from google_oauth import generative_service_client
from enums import EntityType
from duckduckgo_search import DDGS, exceptions
from icecream import ic
from sentence_transformers import util
import hashlib

create_file_id = lambda file_name: py_.chain(
    file_name.lower().split(' ')
).apply_if(
    lambda tokens: tokens[0],
    lambda x: len(x) == 1
).apply_if(
    lambda tokens: py_.join(tokens, '-'),
    lambda x: len(x) < 5 
).apply_if(
    lambda tokens: hashlib.sha256(py_.join(tokens, ' ').encode('utf-8')).hexdigest(),
    lambda x: len(x) > 5 
).value()

gen_datetime_name = (
    lambda: f"{time.strftime('%Y%m%d%H%M%S')}{int((time.time() - int(time.time())) * 1000):03d}"
)

def create_random_dir(parent='.'):
    new_cache_dir = f"{parent}/{gen_datetime_name()}"
    Path(new_cache_dir).mkdir(parents=True, exist_ok=True)

    return Path(new_cache_dir)

def upload_pdfs(files, new_cache_dir=create_random_dir('temp/pdfs/')):
    for file in files:
        file_id = create_file_id(file.name.replace('.pdf',''))
        with open(new_cache_dir.joinpath(file_id).with_suffix('.pdf'), "wb") as f:
            f.write(file.getbuffer())
    return new_cache_dir

def download_pdfs(urls, new_cache_dir=create_random_dir('temp/pdfs/')):
    new_cache_dir = create_random_dir('temp/pdfs/')
    to_load = {
        parse.urlparse(url).path.split("/")[-1].replace(".pdf", ""): url
        for url in urls
    }
    for id, url in to_load.items():
        request.urlretrieve(url, f"{new_cache_dir}/{id}.pdf")
    
    return new_cache_dir

def to_new_cache(files):
    new_cache_dir = create_random_dir('temp/pdfs/')
    for pdf_path in files:
        shutil.move(pdf_path, f"{new_cache_dir}/")

    # remove empty cache dirs
    for folder in Path('temp/pdfs').glob("**/*"):
        if folder.samefile(new_cache_dir): continue
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()

    return new_cache_dir

def uri_validator(x):
    try:
        result = parse.urlparse(x)
        return all([result.scheme, result.netloc])
    except AttributeError:
        return False

def load_pdfs(papers):
    if len(papers) < 1:
        return None, None

    pdf_dir = Path("temp/pdfs")
    PDF_URL_MAP = {}
    for paper in papers:
        if uri_validator(paper):
            PDF_URL_MAP[parse.urlparse(paper).path.split("/")[-1].replace(".pdf", "")] = paper
        else:
            PDF_URL_MAP[create_file_id(paper.name.replace('.pdf',''))] = paper

    CACHED_XMLS = {
        x.name.replace(".tei.xml", ""): x
        for x in Path("temp/xmls").glob("**/*.tei.xml")
    }
    
    to_load = set(PDF_URL_MAP.keys())
    to_process = to_load.difference(CACHED_XMLS.keys())

    ic(to_load)

    if len(to_process) < 1:
        return None, {k: CACHED_XMLS[k] for k in to_load}


    CACHED_PDFS = {
        x.name.replace(".pdf", ""): x for x in Path(pdf_dir).glob("**/*.pdf")
    }

    to_move = py_.objects.omit_by(CACHED_PDFS, lambda _, k: k in CACHED_XMLS.keys())

    new_cache_dir = to_new_cache(to_move.values())
    
    to_load = to_process.difference(to_move.keys())

    if len(to_load) < 1:
        return new_cache_dir, CACHED_XMLS

    to_download = [ PDF_URL_MAP[id] for id in to_load if uri_validator(PDF_URL_MAP[id]) ]
    to_upload = [ PDF_URL_MAP[id] for id in to_load if not uri_validator(PDF_URL_MAP[id]) ]

    if len(to_download) > 0:
        new_cache_dir = download_pdfs(to_download, new_cache_dir)

    if len(to_upload) > 0:
        new_cache_dir = upload_pdfs(to_upload, new_cache_dir)


    return new_cache_dir, CACHED_XMLS



def pdfs_to_xmls(pdfs_folder_name, CACHED_XMLS={}):
    grobid_version = os.environ.get('GROBID_VERSION')
    if pdfs_folder_name == None:
        return CACHED_XMLS

    xmls_folder_name = f"./temp/xmls/{gen_datetime_name()}"
    Path(xmls_folder_name).mkdir(exist_ok=True, parents=True)
    try:
        os.system(f'java -Xmx4G -jar grobid-{grobid_version}/grobid-core/build/libs/grobid-core-{grobid_version}-onejar.jar -gH grobid-{grobid_version}/grobid-home -dIn {pdfs_folder_name} -dOut {xmls_folder_name} -exe processFullText')
        os.system('clear')
        return {
            **CACHED_XMLS,
            **{
                v.name.replace(".tei.xml", ""): v
                for v in Path(xmls_folder_name).glob("*.tei.xml")
            },
        }
    except:
        Path(xmls_folder_name).rmdir()

def xml_to_body_text(xml_path):
    with open(xml_path, "r") as f:
        paper = bs4(f, features="xml")

    [x.decompose() for x in paper.body.select("ref, figure, note")]

    return re.sub("\s+", " ", paper.body.get_text("\n", True)).strip()


prepare_grounding_passages = lambda docs: glm.GroundingPassages(
    passages=py_.chain(docs)
    .map_(lambda doc: glm.Content(parts=[glm.Part(text=doc)]))
    .map_(lambda passage, index: glm.GroundingPassage(content=passage, id=f"{index}"))
    .value()
)

def prepare_corpus(chunks, keywords, regex=True):
    isKeywordInChunk = lambda chunk, keyword: re.search(
        keyword if regex else re.escape(keyword), chunk, re.IGNORECASE
    )
    isChunkUseful = lambda chunk: py_.some(
        keywords, py_.partial(isKeywordInChunk, chunk)
    )
    return py_.chain(chunks).filter_(isChunkUseful).value()


def resolve_hit_documents(corpus, query_hits):
    indices_filtered_corpus = (
        py_.chain(query_hits)
        .flatten()
        .map_(lambda hit: hit["corpus_id"])
        .map_(int)
        .value()
    )
    return (
        py_.chain(corpus)
        .filter_(lambda _, index: index in indices_filtered_corpus)
        .value()
    )


prepare_embeddings = lambda texts: embedder.encode(texts, convert_to_tensor=True)


prepare_query_content = py_.flow(
    lambda user_query: glm.Part(text=user_query), lambda part: glm.Content(parts=[part])
)

LLM_query = (
    lambda entity_type, add_to_query="": {
        EntityType.DATASET: """Extract all named datasets used or mentioned in the provided passages from a research paper as it is.
Do not change or modify the extracted dataset.
Please ensure that the output is in csv format and that only datasets with explicit names are included from the passages.
For clarity, a dataset refers to a collection of organized data points or records that serve a specific purpose.
Datasets are commonly utilized in various fields such as science, research, machine learning, statistics, economics, and more.
They can be structured or unstructured and are often referenced in research papers to support findings, validate hypotheses, or provide evidence for arguments.
Datasets may be explicitly mentioned within the passages, such as "We utilize the <Dataset> collected from <Source> for our analysis." or "The <Dataset> provided by <Provider> contains valuable information for our research."
Additionally, datasets can be constructed from other datasets through aggregation, transformation, or combination processes.
For instance, "We constructed our dataset by merging data from multiple sources, including <Dataset1> and <Dataset2>."
In some cases, the word "dataset" may be implicit, and datasets may be referred to by other terms such as "data collection", "data source", or "data repository".
Ensure that the extraction process accounts for variations in terminology and identifies datasets based on context and proximity to related terms.
Do not consider datasets having "=" in the name.
Ensure that the extraction process focuses on identifying datasets with specific names and excludes general descriptions of data sources or collections. Datasets are alphanumeric words that may not have any meaning.
""",
        EntityType.BASELINE: """Extract all baselines mentioned in the provided passages from a research paper. Please ensure that the output is comma-separated.
For clarity, baselines are established methods or models that serve as benchmarks for comparison when evaluating the performance of new methods, models, or algorithms.
Baselines may be introduced in sentences involving comparisons with other methods or models. Look for keywords such as "compared to", "compared with", "against" or "versus" in sentences discussing comparison methods.
Additionally, baselines are often referenced in the context of established methods or models that are well-known within the research domain. Look for phrases such as "standard method", "traditional approach", "well-known model" or "established baseline" in sentences discussing comparison methods.
Authors may explicitly identify certain methods or models as baselines for comparison purposes. Look for sentences where authors state that a particular method or model is being used as a baseline for evaluating the performance of new approaches. Do not extract common phrases that are not baselines, examples are "state of art", author's name (et al.) and just "basline".
Baselines may also be described in sentences that discuss the implementation, parameters, or assumptions of the baseline approaches. Look for sentences that provide descriptions or explanations of the baseline methods or models being used for comparison.
In the results section of research papers, baselines are typically mentioned in sentences that compare the performance of different methods or models. Look for sentences that present comparative results and discuss how the performance of the proposed approach compares to that of baselines.
If no baselines are found in the provided passages, please return None.
""",
    }[entity_type]
    + add_to_query
)


generate_answer = py_.flow(
    lambda grounded_passages, query_content, temperature=None: glm.GenerateAnswerRequest(
        model="models/aqa",
        contents=[query_content],
        inline_passages=grounded_passages,
        temperature=temperature,
        answer_style="EXTRACTIVE",  # or ABSTRACTIVE, EXTRACTIVE, VERBOSE
    ),
    generative_service_client.generate_answer,
)

regex_keywords_phrases = {
    EntityType.DATASET: [
        "data(set|base)",
        "anal(ytics|ysis)",
        "resear(ch|ch paper)",
        "stud(y|ies?)",
        "exper(iment|iments?)",
        "method(ology|ologies?)",
        "collect(ion|ions?)",
        "sampl(e|ing)",
        "variabl(e|es?)",
        "observ(ation|ations?)",
        "surve(y|ys?)",
        "popul(ation|ations?)",
        "repositor(y|ies?)",
        "databas(e|es?)",
        "sourc(e|es?)",
        "raw data",
        "secondar(y|ies?)",
        "primar(y|ies?)",
        "min(e|ing)",
        "proces(s|sing)",
        "clean(ing|)",
        "manipul(ation|ations?)",
        "integrat(e|ion)",
        "aggregat(e|ion)",
        "visualiz(e|ation)",
        "interpret(ation|ations?)",
        "(used|employed|utilized) for (analysis|modeling|evaluation|research)",
        "(trained|experimented) on",
        "analy(zed|sis) (data|dataset)",
        "(examined|derived|investigated|explored) (data|dataset)",
        "(employed|modeled) with (data|dataset)",
        "(evaluated|tested|compared) on",
        "(referenced|applied) (dataset|data)",
        "(accessed|reviewed) (data|dataset) from",
        "data(-|\s)?set",
        "task",
        "challenge",
        "(knowledge|data)\s*base",
        "benchmark",
        "(experiment|train|performance)[\sa-zA-Z0-9]+on",
        "corpus",
        "class",
        "(train|test)[\sa-zA-Z0-9]+(set)?",
    ],
    EntityType.BASELINE: [
        "compared (to|with)",
        "versus",
        "against",
        "in contrast to",
        "as opposed to",
        "evaluation",
        "assessment",
        "compar(ison|ing|e)",
        "benchmark",
        "reference",
        "outperform",
        "baseline",
        "(standard|traditional|established) (method|model)",
        "(benchmark|reference) (algorithm|model)",
        "(control|prior) method",
        "performance",
        "accuracy",
        "(effectiveness|efficiency|superiority|improvement)",
        "(experimental )?(setup|design|protocol)",
    ],
}

queries = {
    EntityType.DATASET: [
        "Data used in the study",
        "Datasets employed for analysis",
        "Data sources referenced",
        "Dataset utilized for research",
        "Data collection methods",
        "Datasets examined in the paper",
        "Data analysis conducted",
        "Datasets referenced in the research",
        "Data sources investigated",
        "Dataset mentioned in the study",
        "Data utilized for analysis",
        "Datasets considered in the research",
        "Data collection procedures",
        "Dataset discussed in the paper",
        "Data sources utilized",
        "Datasets referenced for analysis",
        "Data used for research purposes",
        "Dataset examined in the study",
        "Data sources referenced in the paper",
        "Datasets employed for investigation",
    ],
    EntityType.BASELINE: [
        "Compare against baselines",
        "Baseline performance evaluation",
        "Benchmark comparison",
        "Reference models assessment",
        "Established method versus",
        "Baseline accuracy comparison",
        "Evaluate against traditional approaches",
        "Benchmark algorithm performance",
        "Control method comparison",
        "Prior method assessment",
        "Compare with standard models",
        "Evaluation protocol for baselines",
        "Baseline experimental setup",
        "Benchmark algorithm effectiveness",
        "Comparison results of baselines",
    ],
}

query_embedder = lambda task_type: prepare_embeddings(queries[task_type])


def verify_entity(entity, entity_type):
    sleep_interval = 1
    query = re.sub("data ?set|corpus|treebank|database|( ){2,}",r"\1",entity)
    match entity_type:
        case EntityType.DATASET:
            query = f'{query} +dataset'
        case EntityType.BASELINE:
            query = f'{query} +baseline'
        case _:
            raise Exception("Entity Type: " + entity_type + " not supported.")

    while True:
        try:
            docs = (
                py_.chain(
                    DDGS().text(
                        query,
                        max_results=5,
                        backend="html",
                    )
                )
                .map_(lambda x: f"{x['title']}: {x['body']}")
                .value()
            )
            if len(docs) < 1:
                return False
            break
        except exceptions.RatelimitException:
            ic('Error: DDGS rate limit exception!')
            time.sleep(sleep_interval)
            sleep_interval *= 1.2
            continue
        except:
            return

    grounding_passages = prepare_grounding_passages(docs)

    query_content = prepare_query_content(f"Is {entity} a data set? (y/n)")

    response = generate_answer(grounding_passages, query_content)
    attempted_answer = py_.attempt(
        lambda _: response.answer.content.parts[0].text, None
    )
    try:
        response = False if attempted_answer == None else "y" in attempted_answer.lower()
        return response
    except:
        ic(attempted_answer)
        return False



def extract_entities(
    chunks, q_embeds, entity_type, keywords=None, entities=set(), verify=True, temperature=None
):
    keywords = keywords or regex_keywords_phrases[entity_type]
    corpus = prepare_corpus(chunks, keywords=keywords, regex=len(entities) < 1)
    corpus_embeds = prepare_embeddings(corpus)

    queries_hits = util.semantic_search(q_embeds, corpus_embeds, top_k=20)
    docs = resolve_hit_documents(corpus, queries_hits)
    grounding_passages = prepare_grounding_passages(docs)

    query_content = py_.flow(LLM_query, prepare_query_content)(
        entity_type,
        (
            "Example {} found are: {}.".format(
                entity_type.lower() + "s", ", ".join(entities)
            )
            if len(entities) > 0
            else ""
        ),
    )

    response = generate_answer(grounding_passages, query_content, temperature)
    attempted_answer = py_.attempt(
        lambda _: response.answer.content.parts[0].text, None
    )

    if py_.is_error(attempted_answer):
        print(attempted_answer)
        return

    for text_in_brackets in re.findall(r"\((.*?)\)", attempted_answer):
        if not re.search(r"\( *(?:[\w& \.,*-]+\d{4};?)+ *\)", text_in_brackets):
            continue
        attempted_answer = re.sub(rf"\({text_in_brackets}\)", "", attempted_answer)

    attempted_answer = sub_ci(" \w+ et\.? al\.", "")(attempted_answer)
    temp_datasets = (
        py_.chain(attempted_answer.split(", "))
        .map_(lambda x: x.strip())
        .filter_(lambda x: len(x.split(" ")) < 10 and "et al." not in x)
        .apply(list)
        .value()
    )
    temp_datasets = entities.union(temp_datasets)

    if temp_datasets - entities == set():
        return entities

    entities = temp_datasets

    if verify:
        entities = set(
            py_.objects.get(
                py_.objects.invert_by({x: verify_entity(x, entity_type) for x in entities}), True
            )
            or []
        )

    if len(entities) == 0:
        return

    entity_keywords = entities

    for dataset in entities:
        if m := re.findall(r"\((.*?)\)", dataset):
            m = [_.strip() for _ in m]
            entity_keywords = entity_keywords.union(m)
            entity_keywords = entity_keywords.union(
                {re.sub(rf"\({_}\)", "", dataset).strip() for _ in m}
            )

    return extract_entities(
        chunks, q_embeds, entity_type, entity_keywords, entities
    )