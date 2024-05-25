import hashlib
import shutil
import time
import typing
from io import BufferedReader
import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib import request
import streamlit as st
import pydash as py_
from bs4 import BeautifulSoup as bs4
from grobid_client.grobid_client import GrobidClient
from enums import TaskType
from CONSTANTS import LLM_TEMPERATURE
from ent_extraction import extract_entities, prepare_embeddings, queries
from models import Downloaded_PDF, Load_XML, Paper, Task, Upload_PDF
from icecream import ic
from texts import chunker, xml_to_body_text
from pathlib import Path
import os
import loguru as lg
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools

T = typing.TypeVar("T")


# @st.cache_data(persist="disk")
@functools.lru_cache
def query_embedder(task_type: TaskType):
    return prepare_embeddings(queries[task_type])


def chcksum(buffer: str | bytes | BufferedReader):
    if isinstance(buffer, str):
        buffer = buffer.encode("utf-8")
    if isinstance(buffer, BufferedReader):
        buffer = buffer.read()
    return hashlib.sha256(buffer).hexdigest()


def check_for_xmls(paper_id: str):
    cached_xmls = {
        path.name.replace(".grobid.tei.xml", ""): path
        for path in Path("temp/xmls").glob("**/*.grobid.tei.xml")
    }

    if paper_id in cached_xmls:
        return cached_xmls[paper_id]

    return None


def load_xml(xml: Load_XML):
    # None path xml already dropped; for precaution
    if xml.path == None or not xml.path.exists():
        return

    with open(xml.path, "r") as file:
        title = bs4(file, features="lxml").title
        if title == None:
            title = xml.id
        else:
            title = py_.human_case(title.get_text())

    return Paper(
        id=xml.id,
        title=title,
        xml_path=xml.path,
        pdf_path={path.stem: path for path in Path("temp/pdfs").glob("**/*.pdf")}[
            xml.id
        ],
    )


gen_datetime_name = (
    lambda: f"{time.strftime('%Y%m%d%H%M%S')}{int((time.time() - int(time.time())) * 1000):03d}"
)


def create_random_dir(parent="."):
    new_cache_dir = f"{parent}/{gen_datetime_name()}"
    Path(new_cache_dir).mkdir(parents=True, exist_ok=True)

    return Path(new_cache_dir)


def upload_pdfs(
    pdfs: py_.chaining.chaining.Chain[list[Upload_PDF]],
    new_cache_dir: Path | None = None,
):
    if new_cache_dir == None:
        new_cache_dir = create_random_dir("temp/pdfs/")

    try:
        for pdf in pdfs.value():
            with open(new_cache_dir.joinpath(pdf.id).with_suffix(".pdf"), "wb") as f:
                f.write(pdf.file.getbuffer())
        return new_cache_dir
    except Exception as e:
        shutil.rmtree(new_cache_dir)
        raise Exception(
            f"{e}\nOccured during uploading PDF {pdf.file.name}. All the PDFs queued for upload will be removed."
        )


def download_pdfs(urls: list[str], new_cache_dir: Path | None = None):
    with TemporaryDirectory(dir=Path("temp")) as tmpdir:
        try:
            for url in urls:
                request.urlretrieve(url, f"{tmpdir}/{chcksum(url)}.pdf")
        except:
            raise Exception(f"Something went wrong, downloading from {url}")

        if new_cache_dir == None:
            new_cache_dir = create_random_dir("temp/pdfs/")

        for pdf in Path(tmpdir).glob("*.pdf"):
            with open(pdf, "rb") as file:
                f_id = chcksum(file)
            # duplicates are overwritten
            shutil.move(
                pdf.rename(pdf.with_name(f_id).with_suffix(".pdf")), new_cache_dir
            )

    return py_.chain(new_cache_dir.glob("*.pdf")).map_(
        lambda pdf: Downloaded_PDF(pdf.name.replace(".pdf", ""), pdf)
    )


def pdfs_to_xmls(pdf_load_dir: Path):
    new_cache_dir = create_random_dir("temp/xmls")

    try:
        client = GrobidClient(config_path="./config.json")
        client.process(
            "processFulltextDocument",
            pdf_load_dir,
            output=new_cache_dir,
            consolidate_header=False,
        )

        return py_.chain(Path(new_cache_dir).glob("**/*.grobid.tei.xml")).map_(
            lambda xml: Load_XML(xml.name.replace(".grobid.tei.xml", ""), xml)
        )
    except Exception as e:
        shutil.rmtree(new_cache_dir)
        shutil.rmtree(pdf_load_dir)
        raise Exception(f"{e}\nOccured during conversion of pdfs to xmls.")


upload_convert = py_.flow(upload_pdfs, pdfs_to_xmls)


def task_wrapper_extract_entities(task: Task):
    start = time.perf_counter()
    q_embeds = query_embedder(task.type)
    lg.logger.debug(f"Query embeddings calculated! {time.perf_counter() - start}s")
    task.extracted_ents = extract_entities(
        verify=task.verify,
        chunks=chunker(task.text),
        temperature=LLM_TEMPERATURE,
        entity_type=task.type,  # type: ignore
        q_embeds=q_embeds,  # type: ignore
    )
    task.time_elapsed = time.perf_counter() - start
    task.pending = False

    return task


def forker(tasks: list[T], function: typing.Callable[[T], T]):
    start = time.perf_counter()

    done__tasks = []

    with mp.Pool(processes=5) as e:
        start = time.perf_counter()
        results = e.map(function, tasks, chunksize=len(tasks) // 5 if len(tasks) > 5 else 1)
        lg.logger.debug(f"Mapped processes! {time.perf_counter() - start}s")
        for result in results:
            done__tasks.append(result)

        # e.shutdown()
    # with multiprocessing.Pool(processes=4) as pool:
    #     results = pool.imap_unordered(function, tasks, 3)

    #     pool.close()
    #     pool.join()

    lg.logger.info("Done!")

    return (done__tasks, time.perf_counter() - start)
