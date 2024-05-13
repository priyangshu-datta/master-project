from collections import namedtuple
from pathlib import Path
import shutil
import time
import pydash as py_
import os
from bs4 import BeautifulSoup as bs4
import re
from urllib import request, parse
import google.ai.generativelanguage as glm
from rsa import verify
import torch
from processing import embedder, sub_ci
from google_oauth import generative_service_client
from enums import EntityType
from duckduckgo_search import DDGS, exceptions
from icecream import ic
from sentence_transformers import util
import hashlib
import requests
from grobid_client.grobid_client import GrobidClient


def create_file_id(file_name):
    tokens = file_name.split(" ")
    if len(tokens) == 1:
        return file_name
    elif len(tokens) < 5:
        return "-".join(tokens)
    else:
        return hashlib.sha256(file_name.encode("utf-8")).hexdigest()


gen_datetime_name = (
    lambda: f"{time.strftime('%Y%m%d%H%M%S')}{int((time.time() - int(time.time())) * 1000):03d}"
)


def create_random_dir(parent="."):
    new_cache_dir = f"{parent}/{gen_datetime_name()}"
    Path(new_cache_dir).mkdir(parents=True, exist_ok=True)

    return Path(new_cache_dir)


def upload_pdfs(pdfs, new_cache_dir=None):
    if new_cache_dir == None:
        new_cache_dir = create_random_dir("temp/pdfs/")
        
    try:
        for pdf in pdfs:
            with open(new_cache_dir.joinpath(hashlib.sha256(pdf.getbuffer()).hexdigest()).with_suffix(".pdf"), "wb") as f:
                f.write(pdf.getbuffer())
        return new_cache_dir
    except Exception as e:
        ic(f"Exception: {e} occured during uploading PDF {pdf.name}. All the PDFs queued for upload will be removed.")
        shutil.rmtree(new_cache_dir)
        return None
        


def download_pdfs(urls, new_cache_dir=None):
    if new_cache_dir == None:
        create_random_dir("temp/pdfs/")
    to_load = {
        parse.urlparse(url).path.split("/")[-1].replace(".pdf", ""): url for url in urls
    }
    for id, url in to_load.items():
        request.urlretrieve(url, f"{new_cache_dir}/{id}.pdf")

    return new_cache_dir


def remove_empty_dirs(parent_dir):
    for folder in parent_dir.iterdir():
        ic(folder.is_dir() and not any(folder.iterdir()))
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def to_new_cache(files):
    new_cache_dir = create_random_dir("temp/pdfs/")
    for pdf_path in files:
        shutil.move(pdf_path, f"{new_cache_dir}/")

    return new_cache_dir


def uri_validator(x):
    try:
        result = parse.urlparse(x)
        return all([result.scheme, result.netloc])
    except AttributeError:
        return False


def getall_pdf_path(pdf_dir):
    return {x.name.replace(".pdf", ""): x for x in pdf_dir.glob("**/*.pdf")}


def load_pdfs(papers):
    
    PDF_path = namedtuple('PDF_path', ['pdf_load_dir', 'xml_cache_dir'])
    
    if len(papers) < 1:
        return PDF_path(None, None)

    pdf_dir = Path("temp/pdfs")
    PDF_URL_MAP = {}
    for paper in papers:
        if uri_validator(paper):
            PDF_URL_MAP[
                parse.urlparse(paper).path.split("/")[-1].replace(".pdf", "")
            ] = paper
        else:
            PDF_URL_MAP[create_file_id(paper.name.replace(".pdf", ""))] = paper

    CACHED_XMLS = {
        x.name.replace(".grobid.tei.xml", ""): x
        for x in Path("temp/xmls").glob("**/*.grobid.tei.xml")
    }

    to_load: set[str] = set(PDF_URL_MAP.keys())
    to_process = to_load.difference(CACHED_XMLS.keys())

    if len(to_process) < 1:
        return PDF_path(None, {k: CACHED_XMLS[k] for k in to_load})

    CACHED_PDFS = getall_pdf_path(pdf_dir)

    to_move = py_.objects.omit_by(CACHED_PDFS, lambda _, k: k in CACHED_XMLS.keys())

    new_cache_dir = to_new_cache(to_move.values())

    to_load = to_process.difference(to_move.keys())

    if len(to_load) < 1:
        remove_empty_dirs(pdf_dir)
        return new_cache_dir, py_.pick_by(
            CACHED_XMLS, lambda _, k: k in PDF_URL_MAP.keys()
        )

    to_download = [PDF_URL_MAP[id] for id in to_load if uri_validator(PDF_URL_MAP[id])]
    to_upload = [
        PDF_URL_MAP[id] for id in to_load if not uri_validator(PDF_URL_MAP[id])
    ]

    if len(to_download) > 0:
        new_cache_dir = download_pdfs(to_download, new_cache_dir)

    if len(to_upload) > 0:
        new_cache_dir = upload_pdfs(to_upload, new_cache_dir)

    remove_empty_dirs(pdf_dir)
    return PDF_path(new_cache_dir, py_.pick_by(CACHED_XMLS, lambda _, k: k in PDF_URL_MAP.keys()))


def pdfs_to_xmls(pdf_load_dir, xmls_path={}):
    if pdf_load_dir == None:
        return xmls_path

    xml_dir = create_random_dir("temp/xmls/")

    pdf_files = [f for f in Path(pdf_load_dir).iterdir() if not f.is_dir()]

    try:
        client = GrobidClient(config_path="./config.json")
        client.process(
            "processFulltextDocument",
            pdf_load_dir,
            output=xml_dir,
            n=len(pdf_files),
            consolidate_header=False,
        )
        
        return {
            **xmls_path,
            **{
                v.name.replace(".grobid.tei.xml", ""): v
                for v in Path(xml_dir).glob("*.grobid.tei.xml")
            },
        }
    except Exception as e:
        ic(f"Exception {e} occued during conversion of pdfs to xmls.")
        shutil.rmtree(xml_dir)
        shutil.rmtree(pdf_load_dir)
        return None

    # grobid_version = os.environ.get('GROBID_VERSION')

    # xmls_folder_name = f"./temp/xmls/{gen_datetime_name()}"

    # try:
    #     os.system(f'java -Xmx4G -jar grobid-{grobid_version}/grobid-core/build/libs/grobid-core-{grobid_version}-onejar.jar -gH grobid-{grobid_version}/grobid-home -dIn {pdf_load_dir} -dOut {xmls_folder_name} -exe processFullText')
    #     os.system('clear')
    #     return {
    #         **CACHED_XMLS,
    #         **{
    #             v.name.replace(".grobid.tei.xml", ""): v
    #             for v in Path(xmls_folder_name).glob("*.grobid.tei.xml")
    #         },
    #     }
    # except:
    #     Path(xmls_folder_name).rmdir()


def xml_to_body_text(xml_path):
    with open(xml_path, "r") as f:
        paper = bs4(f, features="xml")

    [x.decompose() for x in paper.body.select("ref, figure, note")]

    return re.sub(r"\s+", " ", paper.body.get_text("\n", True)).strip()


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
        r"data(set|base)",
        r"anal(ytics|ysis)",
        r"resear(ch|ch paper)",
        r"stud(y|ies?)",
        r"exper(iment|iments?)",
        r"method(ology|ologies?)",
        r"collect(ion|ions?)",
        r"sampl(e|ing)",
        r"variabl(e|es?)",
        r"observ(ation|ations?)",
        r"surve(y|ys?)",
        r"popul(ation|ations?)",
        r"repositor(y|ies?)",
        r"databas(e|es?)",
        r"sourc(e|es?)",
        r"raw data",
        r"secondar(y|ies?)",
        r"primar(y|ies?)",
        r"min(e|ing)",
        r"proces(s|sing)",
        r"clean(ing|)",
        r"manipul(ation|ations?)",
        r"integrat(e|ion)",
        r"aggregat(e|ion)",
        r"visualiz(e|ation)",
        r"interpret(ation|ations?)",
        r"(used|employed|utilized) for (analysis|modeling|evaluation|research)",
        r"(trained|experimented) on",
        r"analy(zed|sis) (data|dataset)",
        r"(examined|derived|investigated|explored) (data|dataset)",
        r"(employed|modeled) with (data|dataset)",
        r"(evaluated|tested|compared) on",
        r"(referenced|applied) (dataset|data)",
        r"(accessed|reviewed) (data|dataset) from",
        r"data(-|\s)?set",
        r"task",
        r"challenge",
        r"(knowledge|data)\s*base",
        r"benchmark",
        r"(experiment|train|performance)[\sa-zA-Z0-9]+on",
        r"corpus",
        r"class",
        r"(train|test)[\sa-zA-Z0-9]+(set)?",
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

import streamlit as st

@st.cache_data(persist="disk")
def query_embedder(task_type: EntityType):
    return prepare_embeddings(queries[task_type])


def verify_counter():
    last_time = time.time()

    def verify_entity(entity, entity_type):

        nonlocal last_time

        sleep_interval = 1

        match entity_type:
            case EntityType.DATASET:
                query = re.sub(
                    "data ?set|corpus|treebank|database|( ){2,}", r"\1", entity
                )
                query = f"{query} +dataset"
            case EntityType.BASELINE:
                query = re.sub("baseline|( ){2,}", r"\1", entity)
                query = f"{query} +baseline"
            case _:
                raise Exception("Entity Type: " + entity_type + " not supported.")

        query = (
            py_.chain(query.split(" "))
            .filter_(lambda tok: len(tok) > 2)
            .apply(lambda x: py_.join(x, " "))
            .value()
        )

        while True:
            try:
                while time.time() - last_time < 0.07:
                    ic("wait!")
                    time.sleep(0.05)
                    continue

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

                last_time = time.time()

                if len(docs) < 1:
                    return False
                break
            except exceptions.RatelimitException as e:
                ic("Error: DDGS rate limit exception!", e)
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
            response = (
                False if attempted_answer == None else "y" in attempted_answer.lower()
            )
            return response
        except:
            ic(attempted_answer)
            return False

    return verify_entity


verify_entity = verify_counter()


def extract_entities(
    chunks: list[str],
    q_embeds: list[torch.Tensor] | torch.Tensor,
    entity_type: EntityType,
    keywords: set[str] | None=None,
    entities: set[str] = set(),
    verify=True,
    temperature: float | None = None,
):
    keywords = keywords or set(regex_keywords_phrases[entity_type])
    corpus = prepare_corpus(chunks, keywords=keywords, regex=len(entities) < 1)
    corpus_embeds = prepare_embeddings(corpus)

    queries_hits = util.semantic_search(q_embeds, corpus_embeds, top_k=20) # type: ignore
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

    ic("Prompting LLM")
    response = generate_answer(grounding_passages, query_content, temperature)
    attempted_answer = py_.attempt(
        lambda _: response.answer.content.parts[0].text, None
    )

    if py_.is_error(attempted_answer):
        print(attempted_answer)
        return

    for text_in_brackets in re.findall(pattern=r"\((.*?)\)", string=attempted_answer): # type: ignore
        if not re.search(r"\( *(?:[\w& \.,*-]+\d{4};?)+ *\)", text_in_brackets):
            continue
        attempted_answer = re.sub(rf"\({text_in_brackets}\)", "", attempted_answer) # type: ignore

    attempted_answer = sub_ci(r" \w+ et\.? al\.", "")(attempted_answer)
    temp_entities = (
        py_.chain(attempted_answer.split(", "))
        .map_(lambda x: x.strip())
        .filter_(lambda x: len(x.split(" ")) < 10 and "et al." not in x)
        .value()
    )

    if verify:
        # using threads will not be helpful due to RateLimitException
        ic("Verifying...")
        temp_entities = set(
            py_.objects.get(
                py_.objects.invert_by(
                    {
                        entity: verify_entity(entity, entity_type)
                        for entity in temp_entities
                    }
                ),
                True,
            )
            or []
        )

    temp_entities = entities.union(temp_entities)

    if temp_entities - entities == set():
        return entities

    entities = temp_entities

    # if len(entities) == 0:
    #     return

    entity_keywords = entities

    for dataset in entities:
        if m := re.findall(r"\((.*?)\)", dataset):
            m = [_.strip() for _ in m]
            entity_keywords = entity_keywords.union(m)
            entity_keywords = entity_keywords.union(
                {re.sub(rf"\({_}\)", "", dataset).strip() for _ in m}
            )

    return extract_entities(
        chunks, q_embeds, entity_type, entity_keywords, entities, verify
    )


# TODO: write a forker to use threading or
# TODO: mulitprocessing to divide the task of
# TODO: entity extraction into multiple
# TODO: process or threads. (Optimization)
# def forker(multple_papers_info):
#
#   // fork_into_mulitple_extract_entities
#
#   return multiple_papers_entites
