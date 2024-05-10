from dotenv import load_dotenv

load_dotenv()

import os
from bootstraps.grobid import grobid_init
from pathlib import Path
import shutil
import utils
from icecream import ic
from pipeline import pdfs_to_text, text_to_entities
import streamlit as st
import pydash as py_
import json
import pandas as pd
from urllib import parse
from streamlit_pdf_viewer import pdf_viewer
from annotate_pdf import annotate_pdf


def main():
    # create the temp folders
    Path("temp/pdfs/").mkdir(exist_ok=True, parents=True)
    Path("temp/xmls/").mkdir(exist_ok=True, parents=True)
    Path("temp/views/").mkdir(exist_ok=True, parents=True)
    Path("temp/transformers/model/").mkdir(exist_ok=True, parents=True)

    # actual code
    # https://arxiv.org/pdf/1705.04304


    df = pd.DataFrame([{"URL": None, "Include": False}])
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True).dropna().query('Include == True')

    files = st.file_uploader("Mulitple uploads allowed", "pdf", True)

    if files or len(set(edited_df["URL"].to_list())) > 0:
        files = files + edited_df["URL"].to_list()
        st.button("Begin Extraction", on_click=lambda: handleClick(files))

    def handleClick(files):
        files = files if files else []
        datasets = []

        if len(files) > 0:
            text_chain = pdfs_to_text(files)
            datasets = text_to_entities(text_chain, verify=True, temperature=0.06)

        CACHED_PDFS = utils.getall_pdf_path(Path("temp/pdfs"))
        tabs = st.tabs(datasets.keys())

        for tab, paper_id in zip(tabs, datasets.keys()):
            with tab:
                pdf_path = CACHED_PDFS[paper_id]
                annotate_pdf(pdf_path, datasets[paper_id])
                st.write(f"Datasets found in paper {paper_id}")
                st.table({"Datasets": datasets[paper_id]})
                pdf_viewer(Path("temp/views").joinpath(pdf_path.name), height=500)


    # remove the temp folders
    # shutil.rmtree('temp')


if __name__ == "__main__":
    main()


# inits = {}

# # initializing grobid
# grobid_version = os.environ.get('GROBID_VERSION')
# if grobid_version == None:
#     raise Exception('No grobid version found in environment.')
# inits['grobid'] = grobid_init(grobid_version)

# if inits['grobid'] == True:
#     ic('GROBID initialized successfully!')
# else:
#     ic('Something went wrong initializing GROBID.')
