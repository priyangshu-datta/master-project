import typing as t
from io import BufferedReader
import re
import time

import google.ai.generativelanguage as glm
import pydash as py_
from duckduckgo_search import DDGS, exceptions
from icecream import ic
from sentence_transformers import util
from torch import Tensor

import loguru as lg

from CONSTANTS import LLM_TEMPERATURE
from enums import TaskType
from goauth import generative_service_client
from texts import embedder, sub_ci
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import hashlib
from bootstrap.chromadb import Chroma


def prepare_grounding_passages(docs: t.List[str]):
    return glm.GroundingPassages(
        passages=py_.chain(docs)
        .map_(lambda doc: glm.Content(parts=[glm.Part(text=doc)]))
        .map_(
            lambda passage, index: glm.GroundingPassage(content=passage, id=f"{index}")
        )
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


def chcksum(buffer):
    if isinstance(buffer, str):
        buffer = buffer.encode("utf-8")
    if isinstance(buffer, BufferedReader):
        buffer = buffer.read()
    return hashlib.sha256(buffer).hexdigest()


# embed_cache = {}


# def prepare_embeddings(texts: list[str]):
#     already_encoded = []
#     not_encoded = []
#     for text in texts:
#         if text in embed_cache:
#             already_encoded.append(embed_cache[chcksum(text)])
#         else:
#             not_encoded.append(text)

#     embeddings = embedder().encode(not_encoded, convert_to_tensor=True)
#     for i, text in enumerate(texts):
#         embed_cache[chcksum(text)] = embeddings[i]
#     return embeddings


def prepare_embeddings(texts: t.List[str]):
    
    return


def prepare_query_content(user_query: str):
    part = glm.Part(text=user_query)
    return glm.Content(parts=[part])


# Ensure that the extraction process accounts for variations in terminology and identifies datasets based on context and proximity to related terms.
def LLM_query(entity_type: TaskType, add_to_query: str):
    return (
        {
            TaskType.DATASET: """Extract all named datasets used or mentioned in the provided passages from a research paper as it is.
Do not change or modify the extracted dataset.
Please ensure that the output is in csv format and that only datasets with explicit names are included from the passages.
For clarity, a dataset refers to a collection of organized data points or records that serve a specific purpose.
Datasets are commonly utilized in various fields such as science, research, machine learning, statistics, economics, and more.
They can be structured or unstructured and are often referenced in research papers to support findings, validate hypotheses, or provide evidence for arguments.
Datasets may be explicitly mentioned within the passages, such as "We utilize the <Dataset> collected from <Source> for our analysis." or "The <Dataset> provided by <Provider> contains valuable information for our research."
Additionally, datasets can be constructed from other datasets through aggregation, transformation, or combination processes.
For instance, "We constructed our dataset by merging data from multiple sources, including <Dataset1> and <Dataset2>."
In some cases, the word "dataset" may be implicit, and datasets may be referred to by other terms such as "data collection", "data source", or "data repository".
Datasets are NOT methods. Methods are something which is applied. Datasets are used on methods. So, extract datasets and ignore methods.
Ensure that the extraction process focuses on identifying datasets with specific names and excludes general descriptions of data sources or collections. Datasets are alphanumeric words that may not have any meaning.
""",
            TaskType.BASELINE: """Extract all baselines mentioned in the provided passages from a research paper. Please ensure that the output is comma-separated.
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


def generate_answer(
    grounding_passages: glm.GroundingPassages,
    query_content: glm.Content,
    temperature: float | None,
):
    answer_request = glm.GenerateAnswerRequest(
        model="models/aqa",
        contents=[query_content],
        inline_passages=grounding_passages,
        temperature=temperature,
        answer_style="EXTRACTIVE",  # or ABSTRACTIVE, EXTRACTIVE, VERBOSE
    )

    return generative_service_client.generate_answer(answer_request)


regex_keywords_phrases = {
    TaskType.DATASET: [
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
    TaskType.BASELINE: [
        r"compared (to|with)",
        "versus",
        "against",
        "in contrast to",
        "as opposed to",
        "evaluation",
        "assessment",
        r"compar(ison|ing|e)",
        "benchmark",
        "reference",
        "outperform",
        "baseline",
        r"(standard|traditional|established) (method|model)",
        r"(benchmark|reference) (algorithm|model)",
        r"(control|prior) method",
        "performance",
        "accuracy",
        r"(effectiveness|efficiency|superiority|improvement)",
        r"(experimental )?(setup|design|protocol)",
    ],
}

queries = {
    TaskType.DATASET: [
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
    TaskType.BASELINE: [
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


def ask_llm(docs: t.List[str], query: str, temperature: None | float):
    grounding_passages = prepare_grounding_passages(docs)

    query_content = prepare_query_content(query)

    return generate_answer(grounding_passages, query_content, temperature)


def search_ddg(query: str):
    docs = []
    sleep_interval = 1
    while True:
        try:
            search_results: t.List[dict[t.Literal["title", "body"], str]] = DDGS().text(
                query, max_results=5, backend="lite"
            )
            docs = (
                py_.chain(search_results)
                .map_(lambda x: f"{x['title']}: {x['body']}")
                .value()
            )

            break
        except exceptions.RatelimitException as e:
            lg.logger.error("Error: DDGS rate limit exception!", e)
            time.sleep(sleep_interval)
            sleep_interval *= 1.2
            if sleep_interval > 60:
                break
            continue
        except:
            break

    return docs


def verify_entity(
    entity: str, entity_type: TaskType, temperature=LLM_TEMPERATURE
) -> t.Tuple[str, t.List[str]]:
    match entity_type:
        case TaskType.DATASET:
            query = re.sub("data ?set|corpus|treebank|database|( ){2,}", r"\1", entity)
            query = f"{query} +dataset"
        case TaskType.BASELINE:
            query = re.sub("baseline|( ){2,}", r"\1", entity)
            query = f"{query} +baseline"
        case _:
            raise Exception("Entity Type: " + entity_type + " not supported.")

    query = (
        py_.chain(query.split(" ")).filter_(lambda tok: len(tok) > 2).join(" ").value()
    )

    docs = search_ddg(query)

    response = ask_llm(docs, f"Is {entity} a data set? (y/n)", temperature)
    attempted_answer = py_.attempt(
        lambda _: response.answer.content.parts[0].text, None
    )

    assert isinstance(attempted_answer, str)

    if "y" in attempted_answer.lower():
        return entity, [
            ga.content.parts[0].text for ga in response.answer.grounding_attributions
        ]
    else:
        return entity, []


def extract_entities(
    chunks: list[str],
    q_embeds: list[Tensor] | Tensor,
    entity_type: TaskType,
    keywords: set[str] | None = None,
    entities: set[str] = set(),
    verify=True,
    temperature: float | None = LLM_TEMPERATURE,
):
    keywords = keywords or set(regex_keywords_phrases[entity_type])
    corpus = prepare_corpus(chunks, keywords=keywords, regex=len(entities) < 1)
    c_embeds = prepare_embeddings(corpus)
    doc_hits = util.semantic_search(q_embeds, c_embeds, top_k=20)  # type: ignore
    docs = resolve_hit_documents(corpus, doc_hits)

    response = ask_llm(
        docs,
        LLM_query(
            entity_type,
            (
                f"Example {entity_type.lower() + 's'} are: {', '.join(entities)}. Find more {entity_type.lower() + 's'}."
                if len(entities) > 0
                else ""
            ),
        ),
        temperature,
    )

    attempted_answer = py_.attempt(
        lambda _: response.answer.content.parts[0].text, None
    )

    assert isinstance(attempted_answer, str), print(attempted_answer)

    intext_citation_regex_1 = re.compile(r"\( *(?:[\w& \.,*-]+\d{4};?)+ *\)")
    intext_citation_regex_2 = re.compile(r" \w+ et\.? al\.")
    inside_bracket_regex = re.compile(r"\((.*?)\)")
    text_in_brackets = re.findall(pattern=inside_bracket_regex, string=attempted_answer)

    for text in text_in_brackets:
        if not re.search(intext_citation_regex_1, text):
            continue
        attempted_answer = re.sub(rf"\({text}\)", "", attempted_answer)

    attempted_answer = sub_ci(intext_citation_regex_2, "")(attempted_answer)
    temp_entities = (
        py_.chain(attempted_answer.split(", "))
        .map_(py_.trim)
        .filter_(lambda x: len(x.split(" ")) < 10 and "et al." not in x.lower())
        .value()
    )

    if verify:
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(verify_entity, temp_entities, repeat(entity_type))

            temp_entities = (
                py_.chain(results)
                .filter_(lambda result: result[1])
                .map_(lambda result: result[0])
                .value()
            )

    temp_entities = entities.union(temp_entities)

    if temp_entities - entities == set():
        lg.logger.info(f"Returning {entity_type}(s).")
        return entities

    entities = entity_keywords = temp_entities

    for entity in entities:
        if match := re.findall(inside_bracket_regex, entity):
            match = [m.strip() for m in match]
            entity_keywords = entity_keywords.union(match)
            entity_keywords = entity_keywords.union(
                {re.sub(rf"\({m}\)", "", entity).strip() for m in match}
            )

    lg.logger.info("Iterative Prompting")
    return extract_entities(
        chunks, q_embeds, entity_type, entity_keywords, entities, verify
    )
