# --- Imports ---
import streamlit as st

from utils.utils import *
from utils.streamlit_ui.ui_components import (
    comp_select_step,
    comp_batch_process_options,
)


# --- Functions ---
def display_step_results(root_dir: str):
    """
    Display the results for a selected step.

    This function prompts the user to select a step from a list of available steps. If requested,
    it calls the 'comp_batch_process_options' function to process the selected step. After processing,
    it displays the results for the selected step.

    Parameters:
        - root_dir (str): The root directory.

    Returns:
        None

    Example:
        display_step_results()
    """

    # Mandatory select step to drilldown
    selected_step = comp_select_step(
        mode="single", sidebar=True, multi_checkbox=False, num_only=True
    )
    str_selected_step = f"step{''.join(selected_step)}"

    # Add batch process component
    comp_batch_process_options(root_dir=root_dir, mode="steps", steps=selected_step)

    # Main view
    if selected_step:
        st.title(f"🔎 {str_selected_step.capitalize()} Drilldown")

        # Placeholder for displaying step-specific benchmarks and comparisons
        # TODO: Implement benchmark visualization
        st.write(f"Benchmarks and comparisons for {str_selected_step} go here.")


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
        page_title="🔎 Steps Drilldown",
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
    display_step_results(PROMPT_ROOT_DIR)


if __name__ == "__main__":
    main()
