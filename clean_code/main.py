from dotenv import load_dotenv

load_dotenv()


import pandas as pd
import streamlit as st
import pydash as py_
from models import Upload_PDF, Load_XML
from util import (
    chcksum,
    check_for_xmls,
    load_xml,
    upload_convert,
    download_pdfs,
    pdfs_to_xmls,
)
from bs4 import BeautifulSoup as bs4
from streamlit.runtime.uploaded_file_manager import UploadedFile

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

    papers_from_upload = upload_convert(to_upload).map_(load_xml).value()

    for paper in papers_from_upload:
        if paper == None:
            continue
        st.session_state.papers[paper.id] = paper


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

    if pdf_cache_dir == None:
        return
    papers_from_download = pdfs_to_xmls(pdf_cache_dir).map_(load_xml).value()

    for paper in papers_from_download:
        if paper == None:
            continue
        st.session_state.papers[paper.id] = paper


def main():
    upload_method, download_method = st.tabs(["Upload", "URL"])

    with upload_method:
        pdfs = st.file_uploader(
            "Upload research papers", accept_multiple_files=True, type="pdf"
        )
        if pdfs != None and len(pdfs) > 0:
            st.button(
                f"Download PDF{'s' if len(pdfs) > 1 else ''}",
                key="upload_btn",
                disabled=st.session_state.disable_upload_btn,
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
        ).dropna()
        if len(edited_df) > 0:
            st.button(
                f"Upload PDF{'s' if len(edited_df) > 1 else ''}",
                key="download_btn",
                disabled=st.session_state.disable_download_btn,
            )

    if st.session_state.disable_upload_btn and pdfs != None:
        load_pdf_uploads(pdfs)
        st.rerun()

    if st.session_state.disable_download_btn and pdfs != None:
        load_pdf_downloads(edited_df["url"].to_list())
        st.rerun()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Entity Mention Extraction", page_icon="⛏️", layout="centered"
    )

    for load in ["upload", "download"]:
        if (
            f"{load}_btn" in st.session_state
            and st.session_state[f"{load}_btn"] == True
        ):
            st.session_state[f"disable_{load}_btn"] = True
        else:
            st.session_state[f"disable_{load}_btn"] = False

    main()
