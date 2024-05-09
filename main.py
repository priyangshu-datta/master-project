from dotenv import load_dotenv
import os
from bootstraps.grobid import grobid_init
from pathlib import Path
import shutil
import utils
from icecream import ic
from pipeline import pdfs_to_text, text_to_entities
import streamlit as st
import pydash as py_


def main():
    # load envs
    load_dotenv()

    inits = {}
    
    # initializing grobid
    grobid_version = os.environ.get('GROBID_VERSION')
    if grobid_version == None:
        raise Exception('No grobid version found in environment.')
    inits['grobid'] = grobid_init(grobid_version)
    
    if inits['grobid'] == True:
        ic('GROBID initialized successfully!')
    else:
        ic('Something went wrong initializing GROBID.')
    
    # create the temp folders
    Path('temp/pdfs/').mkdir(exist_ok=True, parents=True)
    Path('temp/xmls/').mkdir(exist_ok=True, parents=True)
    Path('temp/transformers/model/').mkdir(exist_ok=True, parents=True)


    # actual code
    files = st.file_uploader("Upload Research Paper","pdf",True)

    st.button('Process', on_click=lambda: handleClick(files))

    def handleClick(files):
        files = files if files else []
        datasets = []
        if len(files) > 0:
            text_chain = pdfs_to_text(files)
            # + ['https://arxiv.org/pdf/1705.04304'])
            datasets = text_to_entities(text_chain, verify=True)

        for paper_id, paper_datasets in datasets.items():
            st.write(f"{paper_id}: {', '.join(paper_datasets)}")

    
    
    # remove the temp folders
    shutil.rmtree('temp')


if __name__ == '__main__':
    main()