# --- Imports ---
import os
import streamlit as st

from utils.utils import *
from utils.streamlit_ui.ui_components import comp_show_md_file, st_setup_logo


# --- Functions ---
def display_home(root_dir: str, md_file_dir: str, search_substring: str = "README"):
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

    logo_path = os.path.join(images_dir, "Thesis_AIStreamline.jpeg")

    # Image
    with st.columns(5)[2]:
        st.image(logo_path)

    # Repo explanation
    for file in os.listdir(md_file_dir):
        if file.lower().endswith(".md") and search_substring in file:
            comp_show_md_file(os.path.join(md_file_dir, file))


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
    
    # Setup logo
    st_setup_logo(REPO_ROOT_DIR)

    display_home(REPO_ROOT_DIR, REPO_ROOT_DIR)


if __name__ == "__main__":
    main()
