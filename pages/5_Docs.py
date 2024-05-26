# --- Imports ---
import os
import streamlit as st

from utils.utils import *
from utils.streamlit_ui.ui_components import comp_show_md_file


# --- Functions ---
def display_docs(root_dir: str, md_file_dir: str, search_substring: str = ""):
    """
    Display the home page of the thesis on LLMs for Product Conception.

    Parameters:
    - root_dir (str): The root directory of the project.
    - md_file_dir (str): The directory containing the Markdown files.
    - search_substring (str, optional): The substring to search for in the Markdown file names. Defaults to "README".

    Returns:
    None

    Raises:
    NotImplementedError: If the file extension is not ".md".

    """
    assets_dir = os.path.join(root_dir, "assets")
    images_dir = os.path.join(assets_dir, "images")

    # Image
    with st.columns(5)[2]:
        st.image(os.path.join(images_dir, "Thesis_AIStreamline.jpeg"))

    # Doc search
    doc_file = st.selectbox(
        "Select a document:",
        options=[
            file for file in os.listdir(md_file_dir) if file.lower().endswith(".md")
        ],
        format_func=lambda x: os.path.basename(x).replace("_", " ").capitalize(),
    )

    if doc_file.lower().endswith(".md") and search_substring in doc_file:
        comp_show_md_file(os.path.join(md_file_dir, doc_file))


# --- Main ---
# Streamlit App
def main():
    """
    Main function for the application.

    This function sets up the app configuration, defines the sidebar navigation, and calls the appropriate display functions based on the selected view.

    Parameters:
        None

    Returns:
        None
    """
    # App config
    st.set_page_config(
        page_title="Thesis on LLMs for Product Conception",
        layout="wide",
        initial_sidebar_state="auto",
        page_icon="💡",
        menu_items={
            "About": r"""

            Thesis on LLMs for Product Conception by Guillermo Matas.

            Contact me at https://www.linkedin.com/in/guillermo-matas-ruiz/

            """,
        },
    )

    display_docs(REPO_ROOT_DIR, DOCS_ROOT_DIR)


if __name__ == "__main__":
    main()
