from dotenv import load_dotenv

load_dotenv()


import streamlit as st

st.set_page_config(
    page_title="Entity Mention Extraction", page_icon="⛏️", layout="centered"
)

import time
from pathlib import Path

import pandas as pd
import pydash as py_
from ent_extraction import prepare_embeddings, queries
from enums import TaskType
from icecream import ic
from models import Load_XML, Paper, Task, Upload_PDF
from streamlit.runtime.uploaded_file_manager import UploadedFile
from texts import chunker, xml_to_body_text
from util import (
    chcksum,
    check_for_xmls,
    download_pdfs,
    forker,
    load_xml,
    pdfs_to_xmls,
    task_wrapper_extract_entities,
    upload_convert,
)

"""

A file can exists:
✅ in memory (upload case) 
   - use py_.uniq to get uniq entires
✅ in session_state (uploaded): st.session_state.upload_pdfs
   - filter pdfs already in st.session_state.upload_pdfs
✅ on disk (uploaded and saved; downloaded)
   - already processed (constraint)
      + if not processed then remove the cached pdf and save the new pdf
   - pdfs already on disk
   - not the job here
   - handle in upload_pdf function
✅ save the pdf and load in disk

"""


@st.cache_data(persist="disk")
def query_embedder(task_type: TaskType):
    return prepare_embeddings(queries[task_type])


def load_pdf_uploads(pdfs: list[UploadedFile]):
    pdfs_chain = (
        py_.chain(pdfs)
        .filter_(lambda pdf: isinstance(pdf, UploadedFile))
        .map_(lambda pdf: Upload_PDF(chcksum(pdf.getbuffer()), pdf))
        .uniq_with(lambda file_1, file_2: file_1.id == file_2.id)
        .filter_(lambda pdf: pdf.id not in st.session_state.papers)
    )

    papers_from_disk = (
        pdfs_chain.map_(lambda pdf: Load_XML(pdf.id, check_for_xmls(pdf.id)))
        .map_(load_xml)
        .value()
    )

    for paper in papers_from_disk:
        if paper == None:
            continue
        st.session_state.papers[paper.id] = paper

    to_upload = pdfs_chain.difference_with(
        st.session_state.papers.keys(),
        lambda upload_pdf, paper_id: upload_pdf.id == paper_id,
    )
    if to_upload.apply(len).value() < 1:
        st.rerun()

    try:
        papers_from_upload = upload_convert(to_upload).map_(load_xml).value()
        for paper in papers_from_upload:
            if paper == None:
                continue
            st.session_state.papers[paper.id] = paper
    except Exception as e:
        st.session_state.errors.append(e)
        st.rerun()


def load_pdf_downloads(urls: list[str]):
    ic(urls)
    downloaded_pdfs = download_pdfs(urls).value()
    pdf_cache_dir = None
    for pdf in downloaded_pdfs:
        if pdf.id in st.session_state.papers:
            pdf.file_path.unlink()
        elif (xml_path := check_for_xmls(pdf.id)) != None:
            pdf.file_path.unlink()
            # papers_from_disk
            st.session_state.papers[pdf.id] = load_xml(
                Load_XML(id=pdf.id, path=xml_path)
            )
        else:
            pdf_cache_dir = pdf.file_path.parent

    if pdf_cache_dir == None or len(list(pdf_cache_dir.glob("*.pdf"))) < 1:
        st.rerun()

    try:
        papers_from_download = pdfs_to_xmls(pdf_cache_dir).map_(load_xml).value()
        for paper in papers_from_download:
            if paper == None:
                continue
            st.session_state.papers[paper.id] = paper
    except Exception as e:
        st.session_state.errors.append(e)
        st.rerun()


def main():
    upload_method, download_method = st.tabs(["Upload", "URL"])

    with upload_method:
        pdfs = st.file_uploader(
            "Upload research papers",
            accept_multiple_files=True,
            type="pdf",
            disabled=st.session_state.disable_load_btn,
        )
        if pdfs != None and len(pdfs) > 0:
            st.button(
                f"Upload PDF{'s' if len(pdfs) > 1 else ''}",
                key="upload_btn",
                disabled=st.session_state.disable_load_btn,
            )

    with download_method:
        edited_df = st.data_editor(
            pd.DataFrame([{"url": None}]),
            column_config={
                "url": st.column_config.LinkColumn(
                    label="URL",
                    width="large",
                    validate=r"^https:\/\/.+$",
                    display_text=r"^https:\/\/.+?\/([^\/]+?)$",
                )
            },
            use_container_width=True,
            disabled=st.session_state.disable_load_btn,
        ).dropna()
        if len(edited_df) > 0:
            st.button(
                f"Download PDF{'s' if len(edited_df) > 1 else ''}",
                key="download_btn",
                disabled=st.session_state.disable_load_btn,
            )

    if st.session_state.disable_load_btn:
        if pdfs != None and len(pdfs) > 0 and st.session_state.upload_btn:
            ic("upload")
            load_pdf_uploads(pdfs)
        if len(edited_df["url"]) > 0 and st.session_state.download_btn:
            ic("download")
            load_pdf_downloads(edited_df["url"].to_list())
        st.rerun()

    if len(st.session_state.errors) > 0:
        for err in st.session_state.errors:
            st.error(err)
        st.session_state.errors = []

    if st.session_state.exec_time != None:
        st.success(f"Total time taken: {st.session_state.exec_time}s")

    for paper in st.session_state.papers.values():
        paper: Paper = paper
        expander = st.expander(py_.human_case(paper.title))

        include = expander.checkbox("Include", value=True, key=f"chkbox-{paper.id}")

        if not include:
            continue

        task_types = expander.multiselect(
            "Entities to extract",
            list(TaskType),
            TaskType.DATASET,
            py_.human_case,
            f"mltislct-{paper.id}",
            "More extractable entities can be added",
        )

        potential_task_ids = {
            task_type: chcksum(f"{paper.id}-{task_type}") for task_type in task_types
        }

        status_colors = ["🟡", "🔴", "🟢"]

        status = {}
        att = {}

        for task_id in potential_task_ids.values():
            if (
                task_id in st.session_state.tasks
                and not (task := st.session_state.tasks[task_id]).pending
            ):
                status[task_id] = status_colors[2]
                att[task_id] = task.time_elapsed
                continue

            status[task_id] = status_colors[0]
            att[task_id] = None

        task_df = pd.DataFrame(
            [
                {
                    "task_type": py_.human_case(task_type),
                    "verify": False,
                    "status": status[task_id],
                    "att": att[task_id],
                }
                for task_type, task_id in potential_task_ids.items()
            ]
        )

        edited_task_df = expander.data_editor(
            task_df,
            column_config={
                "task_type": st.column_config.TextColumn(
                    label="Extractable Entities", disabled=True, width="large"
                ),
                "verify": st.column_config.CheckboxColumn(
                    label="Verify Entities", default=True
                ),
                "status": st.column_config.TextColumn(label="Status", disabled=True),
                "att": st.column_config.TextColumn(
                    label="ATT", disabled=True, help="Actual Time Taken"
                ),
            },
            hide_index=True,
            key=f"data-ed-{paper.id}",
            use_container_width=True,
        )

        for row in range(len(edited_task_df)):
            ID = potential_task_ids[edited_task_df["task_type"][row].lower()]
            st.session_state.tasks[ID] = Task(
                paper=paper,
                id=ID,
                task_type=edited_task_df["task_type"][row].lower(),
                verify=edited_task_df["verify"][row],
                chunks=chunker(xml_to_body_text(paper.xml_path)),
            )

    if (
        py_.chain(st.session_state.tasks.values())
        .filter_(lambda task: task.pending)
        .apply(len)
        .value()
        > 0
    ):
        st.button(
            "Extract Entities",
            key="extract_btn",
            disabled=st.session_state.disable_extract_btn,
        )

    if st.session_state.disable_extract_btn:
        tasks: list[Task] = list(st.session_state.tasks.values())
        updated_tasks, exec_time = forker(tasks, task_wrapper_extract_entities)
        for task in updated_tasks:
            st.session_state.tasks[task.id] = task

        st.session_state.exec_time = exec_time
        st.rerun()


if __name__ == "__main__":
    Path("temp/pdfs/").mkdir(exist_ok=True, parents=True)
    Path("temp/xmls/").mkdir(exist_ok=True, parents=True)
    for entity_type in TaskType:
        Path(f"temp/views/{entity_type}").mkdir(exist_ok=True, parents=True)
    Path("cache/transformers/model/").mkdir(exist_ok=True, parents=True)

    if (
        "upload_btn" in st.session_state and st.session_state["upload_btn"] == True
    ) or (
        "download_btn" in st.session_state and st.session_state["download_btn"] == True
    ):
        st.session_state["disable_load_btn"] = True
    else:
        st.session_state["disable_load_btn"] = False

    if "extract_btn" in st.session_state and st.session_state["extract_btn"] == True:
        st.session_state["disable_extract_btn"] = True
    else:
        st.session_state["disable_extract_btn"] = False

    if "papers" not in st.session_state:
        st.session_state.papers = {}

    if "errors" not in st.session_state:
        st.session_state.errors = []

    if "tasks" not in st.session_state:
        st.session_state.tasks = {}
    else:
        st.session_state.tasks = (
            py_.chain(st.session_state.tasks.items())
            .filter_(lambda task: task[1].pending == False)
            .from_pairs()
            .value()
        )

    if "exec_time" not in st.session_state:
        st.session_state.exec_time = None

    ic.configureOutput(prefix=lambda: "%i |> " % int(time.time()))

    main()
