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


def main():
    # create the temp folders
    Path('temp/pdfs/').mkdir(exist_ok=True, parents=True)
    Path('temp/xmls/').mkdir(exist_ok=True, parents=True)
    Path('temp/transformers/model/').mkdir(exist_ok=True, parents=True)

    # actual code
    # https://arxiv.org/pdf/1705.04304

    def handleClick(files):
        files = files if files else []
        datasets = []

        ic(files)

        if len(files) > 0:
            text_chain = pdfs_to_text(files)
            datasets = text_to_entities(text_chain, verify=True, temperature=0.06)

        for paper_id, paper_datasets in datasets.items():
            st.write(f"{paper_id}: {', '.join(paper_datasets)}")


    df = pd.DataFrame([{"URL": None}])
    edited_df = st.data_editor(df, num_rows="dynamic")

    if py_.chain(set(edited_df['URL'].to_list())).filter_(lambda x: x!=None).apply(len).value() > 0:    
        st.button('Process', on_click=lambda: handleClick(edited_df['URL'].to_list()), key='url_btn')
    
    files = st.file_uploader("Upload Research Paper","pdf",True)

    if files:
        st.button('Process', on_click=lambda: handleClick(files), key='file_btn')
    
    
    # remove the temp folders
    # shutil.rmtree('temp')


if __name__ == '__main__':
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
    