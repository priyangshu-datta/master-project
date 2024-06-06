from functools import partial
from threading import Lock

import streamlit as st

from settings import *

st.set_page_config(
    page_title="Entity Mention Extraction", page_icon="⛏️", layout="centered"
)

import time
from pathlib import Path

import pandas as pd
from icecream import ic
from streamlit.runtime.uploaded_file_manager import UploadedFile

from helper.annotate_pdf import annotate_pdf
from helper.models import Load_XML, Paper, Task, TasksBatchDone, Upload_PDF
from helper.texts import clean_text, create_sub_papers, xml_to_body_text
from helper.utils import (chcksum, check_for_xmls, download_pdfs, load_xml,
                   pdfs_to_xmls, upload_convert)
from helper.worker import forker, task_wrapper_extract_entities


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
    st.title("Entity Extractor", help="Extract entities from scientific articles")

    upload_method, download_method = st.columns(2)

    # upload_method, download_method = st.tabs(["Upload", "Download"])

    with upload_method:
        with st.container(border=True):
            st.info("Upload Research Papers")
            pdfs = st.file_uploader(
                "Upload research papers",
                accept_multiple_files=True,
                type="pdf",
                disabled=st.session_state.disable_load_btn
                or st.session_state.disable_extract_btn,
                label_visibility="collapsed",
            )
            if pdfs != None and len(pdfs) > 0:
                st.button(
                    f"Upload PDF{'s' if len(pdfs) > 1 else ''}",
                    key="upload_btn",
                    disabled=st.session_state.disable_load_btn
                    or st.session_state.disable_extract_btn,
                )

    with download_method:
        with st.container(border=True):
            st.info("Download Research Papers")
            edited_df = st.data_editor(
                pd.DataFrame([{"url": None}]),
                column_config={
                    "url": st.column_config.LinkColumn(
                        label="URL",
                        width="medium",
                        validate=r"^https:\/\/.+$",
                        display_text=r"^https:\/\/.+?\/([^\/]+?)$",
                    )
                },
                use_container_width=True,
                disabled=st.session_state.disable_load_btn
                or st.session_state.disable_extract_btn,
                num_rows="dynamic"
            ).dropna()
            if len(edited_df) > 0:
                st.button(
                    f"Download PDF{'s' if len(edited_df) > 1 else ''}",
                    key="download_btn",
                    disabled=st.session_state.disable_load_btn
                    or st.session_state.disable_extract_btn,
                )

    if st.session_state.disable_load_btn:
        if pdfs != None and len(pdfs) > 0 and st.session_state.upload_btn:
            load_pdf_uploads(pdfs)
        if len(edited_df["url"]) > 0 and st.session_state.download_btn:
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

        task_tab, result_tab = expander.tabs(["Tasks", "Results"])

        with task_tab:
            task_df = pd.DataFrame(
                [
                    {
                        "task_type": py_.human_case(task_type),
                        "verify": False,
                        "include": False,
                    }
                    for task_type in TaskType
                ]
            )

            edited_task_df = task_tab.data_editor(
                task_df,
                column_config={
                    "task_type": st.column_config.TextColumn(
                        label="Extractable Entities",
                        disabled=True,
                        width="large",
                        help="Available Entity Types for extraction",
                    ),
                    "verify": st.column_config.CheckboxColumn(
                        label="Verify Entities",
                        default=True,
                        help="Should be verified with internet?",
                    ),
                    "include": st.column_config.CheckboxColumn(
                        label="Include",
                        default=False,
                        help="Should be included in the next extraction?",
                    ),
                },
                hide_index=True,
                key=f"data-ed-{paper.id}",
                use_container_width=True,
                disabled=st.session_state.disable_extract_btn,
            ).query("include==True")

            for row in edited_task_df.T:
                ID = chcksum(f"{paper.id}-{edited_task_df["task_type"][row].lower()}") # type: ignore

                st.session_state.pending_tasks[ID] = Task(
                    paper=paper,
                    id=ID,
                    type=edited_task_df["task_type"][row].lower(),  # type: ignore
                    verify=edited_task_df["verify"][row],  # type: ignore
                    text=xml_to_body_text(paper.xml_path),
                )

        with result_tab:
            result_container = result_tab.container(height=520)
            with result_container:

                done_tasks: list[TasksBatchDone] = st.session_state.done_tasks

                relevant_batch = (
                    py_.chain(done_tasks)
                    .filter_(
                        lambda done_task: paper.id
                        in [r.paper.id for r in done_task.results]
                    )
                    .value()
                )

                for batch in relevant_batch:

                    batch_container = result_container.container(border=True)

                    with batch_container:
                        st.write(f"Batch Completed in {batch.exec_time}s.")

                        annotaions: t.Dict[TaskType, t.Set[str]] = {}

                        for task in batch.results:
                            if task.paper.id != paper.id:
                                continue
                            if task.type in annotaions:
                                annotaions[task.type].union(task.extracted_ents)
                            else:
                                annotaions[task.type] = task.extracted_ents

                        if len(py_.flatten(list(annotaions.values()))) > 1:
                            st.download_button(
                                "Download PDF with compiled Annotations",
                                data=annotate_pdf(
                                    paper.pdf_path,
                                    annotaions,
                                ),
                                file_name=f"{py_.title_case(task.paper.title).replace(' ','_')}_all_extracted.pdf",
                                key=f"{batch.batch_id}-{paper.id}",
                            )

                        for task in batch.results:
                            if task.paper.id != paper.id:
                                continue
                            with batch_container.container(border=True):
                                st.subheader(
                                    f"{py_.human_case(task.type)} Mention Extraction: {task.id}",
                                    divider=True,
                                )
                                if task.verify:
                                    st.write("🟢 Verified with Internet.")
                                else:
                                    st.write("🟡 Not Verified with Internet.")

                                st.write(
                                    "**Actual Time Taken:** {att:.2f}s".format(
                                        att=task.time_elapsed
                                    )
                                )

                                if len(task.extracted_ents) < 1:
                                    st.write(f"No {task.type}s found.")
                                    continue

                                st.dataframe(
                                    pd.DataFrame(
                                        list(task.extracted_ents),
                                        columns=[f"Extracted {task.type}s"],
                                    ),
                                    hide_index=True,
                                    use_container_width=True,
                                )

                                st.download_button(
                                    "Download PDF with Annotations",
                                    data=annotate_pdf(
                                        task.paper.pdf_path,
                                        {task.type: task.extracted_ents},
                                    ),
                                    file_name=f"{py_.title_case(task.paper.title).replace(' ','_')}_{task.type}_extracted.pdf",
                                    key=task.id,
                                )

    if len(st.session_state.pending_tasks) > 0:
        entity_type = "Entitie"
        for task_type in TaskType:
            if all(
                [
                    task.type == task_type
                    for task in st.session_state.pending_tasks.values()
                ]
            ):
                entity_type = py_.human_case(task_type)
        st.button(
            f"Extract {entity_type}s",
            key="extract_btn",
            disabled=st.session_state.disable_extract_btn,
        )

    if st.session_state.disable_extract_btn:
        tasks: t.List[Task] = list(st.session_state.pending_tasks.values())


        modified_tasks: t.List[Task] = []
        for task in tasks:
            text = xml_to_body_text(task.paper.xml_path)
            sub_papers = create_sub_papers(text)

            if len(sub_papers) < 2:
                modified_tasks.append(task)
                continue

            for sub_task_i, sub_paper in enumerate(sub_papers):
                text_span = (
                    clean_text(text).find(sub_paper),
                    clean_text(text).find(sub_paper) + len(sub_paper),
                )
                sub_task = Task(
                    id=chcksum(f"{task.paper.id}-{task.type}-{sub_task_i}"),
                    paper=task.paper,
                    type=task.type,
                    verify=task.verify,
                    text=clean_text(text)[text_span[0] : text_span[1]],
                    parent_task_id=task.id,
                )
                modified_tasks.append(sub_task)

        updated_tasks, exec_time = forker(modified_tasks, partial(task_wrapper_extract_entities, verify_lock=Lock()))
        
        lg.info(updated_tasks)

        done__tasks: t.Dict[str, Task] = {}
        for ut in updated_tasks:
            p_id = ut.parent_task_id
            if p_id == None:
                done__tasks[ut.id] = ut
            else:
                if p_id in done__tasks:
                    done__tasks[p_id].extracted_ents.union(ut.extracted_ents)
                    done__tasks[p_id].time_elapsed = max(
                        ut.time_elapsed, done__tasks[p_id].time_elapsed
                    )
                else:
                    done__tasks[p_id] = Task(
                        p_id,
                        ut.paper,
                        ut.type,
                        ut.verify,
                        time_elapsed=ut.time_elapsed,
                        extracted_ents=ut.extracted_ents,
                        pending=False,
                        text="",
                    )

        st.session_state.done_tasks = [
            *st.session_state.done_tasks,
            TasksBatchDone(
                batch_id=chcksum("-".join([ut.id for ut in done__tasks.values()])),
                exec_time=exec_time,
                results=list(done__tasks.values()),
            ),
        ]

        st.session_state.exec_time = exec_time
        st.rerun()


if __name__ == "__main__":
    ic.configureOutput(prefix=lambda: "%i |> " % int(time.time()))

    Path("temp/pdfs/").mkdir(exist_ok=True, parents=True)
    Path("temp/xmls/").mkdir(exist_ok=True, parents=True)
    # for entity_type in TaskType:
    #     Path(f"temp/views/{entity_type}").mkdir(exist_ok=True, parents=True)
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

    # if "pending_tasks" not in st.session_state:
    st.session_state.pending_tasks = {}

    if "done_tasks" not in st.session_state:
        st.session_state.done_tasks = []

    if "exec_time" not in st.session_state:
        st.session_state.exec_time = None

    main()


# A file can exists:
# ✅ in memory (upload case)
#    - use py_.uniq to get uniq entires
# ✅ in session_state (uploaded): st.session_state.upload_pdfs
#    - filter pdfs already in st.session_state.upload_pdfs
# ✅ on disk (uploaded and saved; downloaded)
#    - already processed (constraint)
#       + if not processed then remove the cached pdf and save the new pdf
#    - pdfs already on disk
#    - not the job here
#    - handle in upload_pdf function
# ✅ save the pdf and load in disk
