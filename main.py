#

import threading
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
import hashlib
import shutil
import utils
import pipeline as pl
import streamlit as st
import pydash as py_
import pandas as pd

# from streamlit_pdf_viewer import pdf_viewer
from annotate_pdf import annotate_pdf
from enums import EntityType
from icecream import ic

# import extra_streamlit_components as stx
from data_classes import Paper, TaskObject
from bs4 import BeautifulSoup as bs4
import time
from error_codes import ERROR_CODES


LLM_TEMPERATURE = 0.06


st.set_page_config(
    page_title="Dataset Mention Extraction",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)


if "load_btn" in st.session_state and st.session_state.load_btn == True:
    st.session_state.disable_load_btn = True
else:
    st.session_state.disable_load_btn = False

if "extract_btn" in st.session_state and st.session_state.extract_btn == True:
    st.session_state.disable_extract_btn = True
else:
    st.session_state.disable_extract_btn = False

if "xmls" not in st.session_state:
    st.session_state.xmls = []

if "warnings" not in st.session_state:
    st.session_state.warnings = {}

if "tasks" not in st.session_state:
    st.session_state.tasks = {}
else:
    st.session_state.tasks = (
        py_.chain(st.session_state.tasks.items())
        .filter_(lambda task: task[1].pending == False)
        .from_pairs()
        .value()
    )


def xml_loader(xml_files):
    xmls = []
    for xml_id, xml_path in xml_files.items():
        title = ""
        with open(xml_path, "r") as file:
            title = bs4(file, features="lxml").title
            if title == None:
                title = xml_id
            else:
                title = py_.human_case(title.get_text())
        xmls.append(Paper(xml_path, title, xml_id))
    return xmls


def safe_thread(function, args):
    t = threading.Thread(target=function, args=args)
    t.start()
    t.join()


def task_wrapper_extract_entities(task: TaskObject):
    begin = time.time()
    task.extracted_ents = utils.extract_entities(
        verify=task.verify,
        chunks=task.chunks,
        temperature=task.temperature,
        entity_type=task.entity_type.lower(),  # type: ignore
        q_embeds=task.query_embeds,  # type: ignore
    )
    end = time.time()
    task.pending = False
    task.time_elapsed = end - begin


def forker(tasks: list[TaskObject], function):
    for task in tasks:
        safe_thread(function, [task])


def chcksum(buffer):
    if isinstance(buffer, str):
        buffer = buffer.encode("utf-8")
    return hashlib.sha256(buffer).hexdigest()


def main():
    files = st.file_uploader(
        "Upload research papers", accept_multiple_files=True, type="pdf"
    )

    st.button("Load PDFs", key="load_btn", disabled=st.session_state.disable_load_btn)

    if st.session_state.disable_load_btn:
        files = (
            py_.chain(files or [])
            .map_(lambda file: (chcksum(file.getbuffer()), file))
            .uniq_with(lambda file_1, file_2: file_1[0] == file_2[0])
            .map_(lambda file: file[1])
        )

        XML_CACHE = [
            xml_p.name.replace(".grobid.tei.xml", "")
            for xml_p in Path("temp/xmls").glob("**/*.grobid.tei.xml")
        ]

        temp_xmls = files.filter_(
            lambda pdf: chcksum(pdf.getbuffer()) not in XML_CACHE
        ).value()

        persisted_xmls = py_.pick(
            {
                xml_p.name.replace(".grobid.tei.xml", ""): xml_p
                for xml_p in Path("temp/xmls").glob("**/*.grobid.tei.xml")
            },
            *files.difference(temp_xmls)
            .map_(lambda file: chcksum(file.getbuffer()))
            .value(),
        )

        if len(temp_xmls) < 1:
            st.session_state.warnings[ERROR_CODES.E01] = "PDFs are already loaded!"
            for xml in xml_loader(persisted_xmls):
                st.session_state.xmls.append(xml)
            st.rerun()

        new_pdf_dir = utils.upload_pdfs(temp_xmls)
        if new_pdf_dir == None:
            st.rerun()
        xml_files = utils.pdfs_to_xmls(new_pdf_dir)
        if xml_files == None:
            st.rerun()

        for xml in xml_loader(xml_files):
            st.session_state.xmls.append(xml)

        st.rerun()

    if len(st.session_state.xmls) > 0 and not st.session_state.disable_load_btn:
        if ERROR_CODES.E01 in st.session_state.warnings:
            st.warning(st.session_state.warnings[ERROR_CODES.E01])
            del st.session_state.warnings[ERROR_CODES.E01]
        else:
            st.success("Succesfully loaded!", icon="✅")

    for xml in st.session_state.xmls:
        expander = st.expander(py_.human_case(xml.title))
        expander.write(xml.paper_id)
        
        include = expander.checkbox("Include", value=True, key=f"chkbox-{xml.paper_id}")

        if not include:
            continue

        tasks = expander.multiselect(
            "Entities to extract",
            list(EntityType),
            EntityType.DATASET,
            py_.human_case,
            f"mltislct-{xml.paper_id}",
            "More extractable entities can be added",
        )

        task_ids = {task: chcksum(f"{xml.paper_id}-{task.lower()}") for task in tasks}

        status_codes = ["🟡", "🔴", "🟢"]

        status = {}
        att = {}

        for task in tasks:
            if (
                task_ids[task] not in st.session_state.tasks
                or st.session_state.tasks[task_ids[task]].pending
            ):
                status[task] = status_codes[0]
                att[task] = None
            elif (
                task_ids[task] in st.session_state.tasks
                and st.session_state.tasks[task_ids[task]].pending == False
            ):
                status[task] = status_codes[2]
                att[task] = st.session_state.tasks[task_ids[task]].time_elapsed

        task_df = expander.data_editor(
            pd.DataFrame(
                [
                    {
                        "task": py_.human_case(task),
                        "verify": False,
                        "status": status[task],
                        "att": att[task],
                    }
                    for task in tasks
                ]
            ),
            column_config={
                "task": st.column_config.TextColumn(
                    label="Extractable Entities", disabled=True, width="large"
                ),
                "verify": st.column_config.CheckboxColumn(
                    label="Verify Entities", default=True
                ),
                "status": st.column_config.TextColumn(label="Status", disabled=True),
                "att": st.column_config.TextColumn(label="ATT", disabled=True),
            },
            hide_index=True,
            key=f"data-ed-{xml.paper_id}",
            use_container_width=True,
        )

        text = utils.xml_to_body_text(xml.path)
        chunks = pl.chunker(text)
                

        for row in task_df.T:
            if task_df["status"][row] == "🟢":  # type: ignore
                continue
            st.session_state.tasks[
                chcksum(f"{xml.paper_id}-{task_df['task'][row].lower()}")  # type: ignore
            ] = TaskObject(
                task_id=task_ids[task_df["task"][row].lower()],  # type: ignore
                paper_id=xml.paper_id,
                entity_type=task_df["task"][row].lower(),  # type: ignore
                verify=task_df["verify"][row],  # type: ignore
                xml_path=xml.path,
                chunks=chunks,
                query_embeds=utils.query_embedder(task_df["task"][row].lower()),  # type: ignore
                temperature=LLM_TEMPERATURE,
            )
        
        

    if len(st.session_state.tasks) > 0:
        st.button(
            "Extract Entities",
            key="extract_btn",
            disabled=st.session_state.disable_extract_btn
            or len(
                py_.chain(st.session_state.tasks.values())
                .filter_(lambda task: task.pending)
                .value()
            )
            < 1,
        )

    if st.session_state.disable_extract_btn:
        
        forker(
            py_.chain(st.session_state.tasks.values())
            .filter_(lambda task: task.pending)
            .value(),
            task_wrapper_extract_entities,
        )

        st.rerun()

    if (
        not st.session_state.disable_extract_btn
        and len(
            py_.chain(st.session_state.tasks.items())
            .filter_(lambda task: task[1].pending)
            .value()
        )
        != 0
    ):
        st.toast("Extracted Successfully", icon="✅")


if __name__ == "__main__":
    Path("temp/pdfs/").mkdir(exist_ok=True, parents=True)
    Path("temp/xmls/").mkdir(exist_ok=True, parents=True)
    for entity_type in EntityType:
        Path(f"temp/views/{entity_type}").mkdir(exist_ok=True, parents=True)
    Path("cache/transformers/model/").mkdir(exist_ok=True, parents=True)
    
    ic.configureOutput(prefix=lambda: '%i |> ' % int(time.time()))

    main()

    # del st.session_state.tasks
    # shutil.rmtree("temp")
