# --- Imports ---
import os
import streamlit as st

from utils.utils import *


# --- Functions ---
def comp_show_md_file(file_path: str):
    """
    Display the contents of a Markdown file.

    This function takes a file path as input and displays the contents of the file if it has a '.md' extension.
    The function reads the file, converts it to Markdown format, and displays it using the 'st.markdown' function.

    Parameters:
        file_path (str): The path of the Markdown file to be displayed.

    Returns:
        None

    Raises:
        NotImplementedError: If the file does not have a '.md' extension.

    Example:
        comp_show_md_file("/path/to/file.md")
    """
    if file_path.endswith(".md"):
        with open(file_path, "r") as f:
            md_file = f.read()
        st.markdown(md_file)
    else:
        raise NotImplementedError


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

    # Image
    with st.columns(5)[2]:
        st.image(os.path.join(images_dir, "Thesis_AIStreamline.jpeg"))

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

    display_home(REPO_ROOT_DIR, REPO_ROOT_DIR)


if __name__ == "__main__":
    main()
