# 

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
from streamlit_pdf_viewer import pdf_viewer
from annotate_pdf import annotate_pdf
from enums import EntityType
from icecream import ic
import extra_streamlit_components as stx
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

# if "entities" not in st.session_state:
#     st.session_state["entities"] = {entity_type: {} for entity_type in EntityType}

# if "research_papers" not in st.session_state:
#     st.session_state["research_papers"] = set()

# if "extract_btn" in st.session_state and st.session_state.extract_btn == True:
#     st.session_state.btn_disable = True
# else:
#     st.session_state.btn_disable = False


# def handleClick(files, verify, entity_types, force_rerun):
#     if len(files) < 1:
#         return

#     pdf_load_dir, xml_cache_dir = utils.load_pdfs(files)

#     if pdf_load_dir != None:
#         xml_cache_dir = utils.pdfs_to_xmls(pdf_load_dir, xml_cache_dir)

#     {
#         paper_id: utils.xml_to_body_text(xml_path)
#         for paper_id, xml_path in xml_cache_dir.items()
#     }

#     taskobject = TaskObject()

#     ic(f"Pre-processing pdfs")
#     text_chain = pl.pdfs_to_text(files)
#     ic(f"Pre-processed pdfs")

#     for entity_type in entity_types:
#         entity_text_chain = text_chain.filter_(
#             lambda id_path: id_path[0]
#             not in list(st.session_state["entities"][entity_type].keys())
#         )

#         if force_rerun:
#             entity_text_chain = text_chain
#         ic("Starting extraction...")
#         st.session_state["entities"][entity_type] = {
#             **pl.text_to_entities(
#                 entity_text_chain,
#                 entity_type,
#                 verify=verify,
#                 temperature=LLM_TEMPERATURE,
#             ),
#             **st.session_state["entities"][entity_type],
#         }
#         ic("Extracted")

#     st.session_state["research_papers"] = (
#         py_.chain(st.session_state["entities"].values())
#         .map_(lambda ent_dict: list(ent_dict.keys()))
#         .flatten()
#         .apply(set)
#         .value()
#     )


if "load_btn" in st.session_state and st.session_state.load_btn == True:
    st.session_state.disable_load_btn = True
else:
    st.session_state.disable_load_btn = False

if "xmls" not in st.session_state:
    st.session_state.xmls = []

if "warnings" not in st.session_state:
    st.session_state.warnings = {}

if "tasks" not in st.session_state:
    st.session_state.tasks = []


def main():
    Path("temp/pdfs/").mkdir(exist_ok=True, parents=True)
    Path("temp/xmls/").mkdir(exist_ok=True, parents=True)
    for entity_type in EntityType:
        Path(f"temp/views/{entity_type}").mkdir(exist_ok=True, parents=True)
    Path("temp/transformers/model/").mkdir(exist_ok=True, parents=True)

    files = st.file_uploader(
        "Upload research papers", accept_multiple_files=True, type="pdf"
    )

    st.button("Load PDFs", key="load_btn", disabled=st.session_state.disable_load_btn)

    if st.session_state.disable_load_btn:
        files = (
            py_.chain(files)
            .filter_(
                lambda pdf: pdf.file_id
                not in [xml.paper_id for xml in st.session_state.xmls]
            )
            .value()
        )

        if len(files) < 1:
            st.session_state.warnings[ERROR_CODES.E01] = "PDFs are already loaded!"
            st.rerun()

        new_pdf_dir = utils.upload_pdfs(files)
        if new_pdf_dir == None:
            st.rerun()
        xml_files = utils.pdfs_to_xmls(new_pdf_dir)
        if xml_files == None:
            st.rerun()

        for xml_id, xml_path in xml_files.items():
            title = ""
            with open(xml_path, "r") as file:
                title = py_.human_case(bs4(file, features="lxml").title.get_text())
            xml = Paper(xml_path, title, xml_id)
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
        task_df = expander.data_editor(
            pd.DataFrame(
                [
                    {
                        "task": py_.human_case(entity_type),
                        "verify": False,
                    }
                    for entity_type in tasks
                ]
            ),
            column_config={
                "Extractable Entities": st.column_config.TextColumn(
                    disabled=True, width="large"
                ),
                "Verify Entities": st.column_config.CheckboxColumn(default=True),
            },
            hide_index=True,
            key=f"data-ed-{xml.paper_id}",
            use_container_width=True,
        )
        

        text = utils.xml_to_body_text(xml.path)
        chunks = pl.chunker(text)

        for row in task_df.T:
            st.session_state.tasks.append(
                TaskObject(
                    paper_id=xml.paper_id,
                    entity_type=task_df["task"][row],
                    verify=task_df["verify"][row],
                    xml_path=xml.path,
                    chunks=chunks,
                    query_embeds=utils.query_embedder(task_df["task"][row].lower()),
                    temperature=LLM_TEMPERATURE,
                )
            )
    

    # shutil.rmtree("temp")
    
    
if __name__ == "__main__":
    main()
