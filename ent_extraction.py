import re
import time

import google.ai.generativelanguage as glm
import pydash as py_
from duckduckgo_search import DDGS, exceptions
from icecream import ic
from sentence_transformers import util
from torch import Tensor

from CONSTANTS import LLM_TEMPERATURE
from enums import TaskType
from goauth import generative_service_client
from texts import embedder, sub_ci
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

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
Ensure that the extraction process accounts for variations in terminology and identifies datasets based on context and proximity to related terms.
Do not consider datasets having "=" in the name.
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


generate_answer = py_.flow(
    lambda grounded_passages, query_content, temperature: glm.GenerateAnswerRequest(
        model="models/aqa",
        contents=[query_content],
        inline_passages=grounded_passages,
        temperature=temperature,
        answer_style="EXTRACTIVE",  # or ABSTRACTIVE, EXTRACTIVE, VERBOSE
    ),
    generative_service_client.generate_answer,
)

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


def verify_counter():
    last_time = time.time()

    def verify_entity(entity: str, entity_type: TaskType, temperature=LLM_TEMPERATURE):

        nonlocal last_time

        sleep_interval = 1

        match entity_type:
            case TaskType.DATASET:
                query = re.sub(
                    "data ?set|corpus|treebank|database|( ){2,}", r"\1", entity
                )
                query = f"{query} +dataset"
            case TaskType.BASELINE:
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
                # while time.time() - last_time < 0.07:
                #     time.sleep(0.05)
                #     continue

                docs = (
                    py_.chain(
                        DDGS().text(
                            query,
                            max_results=5,
                        )
                    )
                    .map_(lambda x: f"{x['title']}: {x['body']}")
                    .value()
                )

                # last_time = time.time()

                if len(docs) < 1:
                    return (entity, False)
                break
            except exceptions.RatelimitException as e:
                ic("Error: DDGS rate limit exception!", e)
                time.sleep(sleep_interval)
                sleep_interval *= 1.2
                continue
            except:
                return (entity, False)
        grounding_passages = prepare_grounding_passages(docs)

        query_content = prepare_query_content(f"Is {entity} a data set? (y/n)")

        response = generate_answer(grounding_passages, query_content, temperature)
        attempted_answer = py_.attempt(
            lambda _: response.answer.content.parts[0].text, None
        )

        assert isinstance(attempted_answer, str)

        try:
            return (
                entity,
                False if attempted_answer == None else "y" in attempted_answer.lower(),
            )

        except:
            # ic(attempted_answer)
            return entity, False

    return verify_entity


verify_entity = verify_counter()


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
    corpus_embeds = prepare_embeddings(corpus)

    queries_hits = util.semantic_search(q_embeds, corpus_embeds, top_k=20)  # type: ignore
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

    ic("Asking LLM")
    response = generate_answer(grounding_passages, query_content, temperature)
    attempted_answer = py_.attempt(
        lambda _: response.answer.content.parts[0].text, None
    )

    if py_.is_error(attempted_answer):
        print(attempted_answer)
        return

    for text_in_brackets in re.findall(pattern=r"\((.*?)\)", string=attempted_answer):  # type: ignore
        if not re.search(r"\( *(?:[\w& \.,*-]+\d{4};?)+ *\)", text_in_brackets):
            continue
        attempted_answer = re.sub(rf"\({text_in_brackets}\)", "", attempted_answer)  # type: ignore

    attempted_answer = sub_ci(r" \w+ et\.? al\.", "")(attempted_answer)
    temp_entities = (
        py_.chain(attempted_answer.split(", "))
        .map_(lambda x: x.strip())
        .filter_(lambda x: len(x.split(" ")) < 10 and "et al." not in x)
        .value()
    )

    if verify:
        ic("Verifying")
        with ThreadPoolExecutor() as executor:
            results = executor.map(verify_entity, temp_entities, repeat(entity_type))

            temp_entities = (
                py_.chain(results)
                .filter_(lambda result: result[1])
                .map_(lambda result: result[0])
                .value()
            )

    temp_entities = entities.union(temp_entities)

    if temp_entities - entities == set():
        return entities

    entities = temp_entities

    entity_keywords = entities

    for dataset in entities:
        if m := re.findall(r"\((.*?)\)", dataset):
            m = [_.strip() for _ in m]
            entity_keywords = entity_keywords.union(m)
            entity_keywords = entity_keywords.union(
                {re.sub(rf"\({_}\)", "", dataset).strip() for _ in m}
            )
    ic("Recurring")
    return extract_entities(
        chunks, q_embeds, entity_type, entity_keywords, entities, verify
    )
