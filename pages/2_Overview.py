# --- Imports ---
import streamlit as st

from utils.utils import *
from utils.streamlit_ui.ui_components import comp_batch_process_options


# --- Functions ---
def display_overview(root_dir: str):
    """
    Display the results overview for all available steps.

    This function lets the user have an oversight of the prompting and benchmarking results.
    If requested, it calls the 'comp_batch_process_options' function to process all steps.
    After processing, them it displays the results for all steps.

    Parameters:
        - root_dir (str): The root directory.

    Returns:
        None
    """
    comp_batch_process_options(root_dir=root_dir, mode="batch")

    st.title("Overview Results")
    # Placeholder for displaying overall benchmark and drilldowns
    # TODO: Implement benchmark visualization
    st.write("Overall benchmarks and detailed drilldowns go here.")


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
        page_title="📊 Overview",
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

    # Display view
    display_overview(PROMPT_ROOT_DIR)


if __name__ == "__main__":
    main()
