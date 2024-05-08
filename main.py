from dotenv import load_dotenv
import os
from bootstraps.grobid import grobid_init
from pathlib import Path
import shutil
import utils
from icecream import ic
from pipeline import url_to_text, text_to_entities


def main():
    # load envs
    load_dotenv()

    inits = {}
    
    # initializing grobid
    grobid_version = os.environ.get('GROBID_VERSION')
    inits['grobid'] = grobid_init(grobid_version)
    
    # create the temp folders
    Path('temp/pdfs/').mkdir(exist_ok=True, parents=True)
    Path('temp/xmls/').mkdir(exist_ok=True, parents=True)


    # actual code

    text_chain = url_to_text([
        ''
    ])

    datasets = text_to_entities(text_chain, verify=True)

    ic(datasets)

    # remove the temp folders
    # shutil.rmtree('temp')


if __name__ == '__main__':
    main()