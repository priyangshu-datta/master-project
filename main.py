from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
import utils
from pipeline import pdfs_to_text, text_to_entities
import streamlit as st
import pydash as py_
import pandas as pd
from streamlit_pdf_viewer import pdf_viewer
from annotate_pdf import annotate_pdf
from enums import EntityType
from icecream import ic
import extra_streamlit_components as stx


LLM_TEMPERATURE = 0.06


st.set_page_config(
    page_title="Dataset Mention Extraction",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)

if "entities" not in st.session_state:
    st.session_state["entities"] = {entity_type: {} for entity_type in EntityType}

if "research_papers" not in st.session_state:
    st.session_state["research_papers"] = set()


def main():
    Path("temp/pdfs/").mkdir(exist_ok=True, parents=True)
    Path("temp/xmls/").mkdir(exist_ok=True, parents=True)
    for entity_type in EntityType:
        Path(f"temp/views/{entity_type}").mkdir(exist_ok=True, parents=True)
    Path("temp/transformers/model/").mkdir(exist_ok=True, parents=True)

    URL_df = pd.DataFrame([{"URL": None, "Include": False}])
    edited_df = (
        st.data_editor(URL_df, num_rows="dynamic", use_container_width=True)
        .dropna()
        .query("Include == True")
    )

    files = st.file_uploader("Mulitple uploads allowed", "pdf", True)

    if files or len(set(edited_df["URL"].to_list())) > 0:
        files = files + edited_df["URL"].to_list()
        entity_types = st.multiselect(
            "Entities to extract", list(EntityType), format_func=py_.human_case
        )
        verify = st.toggle("Verify entities", value=True)
        st.button(
            "Begin Extraction",
            on_click=lambda: handleClick(files, verify, entity_types),
        )

    def handleClick(files, verify, entity_types):
        files = files if files else []

        if len(files) < 1:
            return

        text_chain = pdfs_to_text(files)

        for entity_type in entity_types:
            entity_text_chain = text_chain.filter_(
                lambda id_path: id_path[0]
                not in list(st.session_state["entities"][entity_type].keys())
            )

            if force_rerun := False:
                entity_text_chain = text_chain
            st.session_state["entities"][entity_type] = {
                **text_to_entities(
                    entity_text_chain,
                    entity_type,
                    verify=verify,
                    temperature=LLM_TEMPERATURE,
                ),
                **st.session_state["entities"][entity_type],
            }

        st.session_state["research_papers"] = (
            py_.chain(st.session_state["entities"].values())
            .map_(lambda ent_dict: list(ent_dict.keys()))
            .flatten()
            .apply(set)
            .value()
        )

    CACHED_PDFS = utils.getall_pdf_path(Path("temp/pdfs"))
    ANNOTATED_CACHED_PDFS = {}
    for entity_type in EntityType:
        ANNOTATED_CACHED_PDFS[entity_type] = utils.getall_pdf_path(
            Path(f"temp/views/{entity_type}")
        )

    if len(st.session_state["research_papers"]) < 1:
        ic(st.session_state["research_papers"])
        return

    with st.container(border=True):

        r_tab = stx.tab_bar(
            data=[
                stx.TabBarItemData(i, r, None)
                for i, r in enumerate(st.session_state["research_papers"])
            ],
            default=0,
            return_type=int,
        )

        with st.container(border=True):

            e_tab = stx.tab_bar(
                data=[
                    stx.TabBarItemData(i, py_.human_case(e), None)
                    for i, e in enumerate(EntityType)
                ],
                default=0,
                return_type=int,
            )
            
            entity_type = list(EntityType)[e_tab]
            paper_id = list(st.session_state["research_papers"])[r_tab]
            
            
            if paper_id not in ANNOTATED_CACHED_PDFS[entity_type]:
                annotate_pdf(
                    CACHED_PDFS[paper_id],
                    entity_type,
                    st.session_state["entities"][entity_type][paper_id],
                )

            if len(st.session_state["entities"][entity_type][paper_id]) < 1:
                st.write(f"No {entity_type}s found in paper {paper_id}.")
            else:
                st.write(
                    f"{py_.human_case(entity_type)} found in paper {paper_id}."
                )
            
                st.table(
                    {
                        f"{py_.human_case(entity_type)}": st.session_state[
                            "entities"
                        ][entity_type][paper_id]
                    }
                )

                pdf_viewer(
                    Path("temp/views")
                    .joinpath(
                        entity_type, paper_id
                    )
                    .with_suffix(".pdf"),
                    height=500
                )

            # st.write(st.session_state["entities"][list(EntityType)[e_tab]][list(st.session_state["research_papers"])[r_tab]])

    # research_paper_tabs = st.tabs(st.session_state["research_papers"])
    # for r_tab, paper_id in zip(
    #     research_paper_tabs, st.session_state["research_papers"]
    # ):
    #     with r_tab:
    #         # entity_tabs = st.tabs(list(map(py_.human_case, list(EntityType))))

    #         for entity_type in EntityType:
    #                 # with e_tab:

    #             st.session_state["current_view"] = f"{paper_id}-{entity_type}"

    #             entities = st.session_state["entities"][entity_type]
    #             if len(entities) < 1:
    #                 continue

                # if paper_id not in ANNOTATED_CACHED_PDFS[entity_type]:
                #     annotate_pdf(
                #         CACHED_PDFS[paper_id],
                #         entity_type,
                #         st.session_state["entities"][entity_type][paper_id],
                #     )

                # if len(st.session_state["entities"][entity_type][paper_id]) < 1:
                #     st.write(f"No {entity_type}s found in paper {paper_id}.")
                #     continue

                # st.write(
                #     f"{py_.human_case(entity_type)} found in paper {paper_id}."
                # )

                # st.table(
                #     {
                #         f"{py_.human_case(entity_type)}": st.session_state[
                #             "entities"
                #         ][entity_type][paper_id]
                #     }
                # )

    #             pdf_viewer(
    #                 Path("temp/views")
    #                 .joinpath(entity_type, paper_id)
    #                 .with_suffix(".pdf"),
    #                 width=700,
    #                 height=500
    #             )

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


# TODO: Break the paper into multiple sub-papers and run the task parallely lastly join the output.
