# --- Imports ---
import os
import json
import streamlit as st
from streamlit_monaco import st_monaco

from utils.utils import *
from utils.streamlit_ui.ui_components import (
    comp_select_step,
    comp_batch_process_options,
)


# --- Functions ---


def display_edit_json_sources(root_dir):
    """
    Display and edit JSON sources.

    Parameters:
    - root_dir (str): The root directory.

    Returns:
    - None

    This function displays a title "Edit Sources" and allows the user to select a source type from a radio group.
    It then prompts the user to select a step from a dropdown menu.
    The function also provides batch process options using the 'comp_batch_process_options' function.
    If a step and source type are selected, the function displays a dropdown menu to select a template file.
    Once a file is selected, the function loads and displays the JSON data using the 'st_monaco' function.
    The user can edit the JSON data and save it by clicking the "Save JSON Prompt" button.

    Note:
    - The 'comp_select_step' function is used to select a step.
    - The 'comp_batch_process_options' function is used to provide batch process options.

    """

    # Define the directory containing the JSON templates
    input_dir = os.path.join(root_dir, "input")
    selected_step = None
    selected_file = None
    selected_step = None

    # Dropdown to select a step
    selected_step = comp_select_step(
        mode="single", sidebar=True, multi_checkbox=False, num_only=True
    )
    str_selected_step = f"step{''.join(selected_step)}"

    # Add batch process options
    comp_batch_process_options(root_dir=root_dir, mode="steps", steps=selected_step)

    # Radio group to select the file type to be modified
    st.title("✏️ Source Editor")
    source_types = ["source", "templates", "expected_response"]
    source_type = st.radio(
        label="Select a source type:",
        options=source_types,
        format_func=lambda x: x.replace("_", " ").capitalize(),
        horizontal=True,
    )

    if str_selected_step and source_type in source_types:
        source_dir = os.path.join(input_dir, str_selected_step, source_type)
        source_files = [f for f in os.listdir(source_dir) if f.endswith(".json")]

        # Dropdown to select a template file
        selected_file = st.selectbox(f"Select {source_type} file:", source_files)

    if selected_file:
        source_path = os.path.join(source_dir, selected_file)

        # Load and display the selected JSON template
        with open(source_path, "r") as file:
            json_data = json.load(file)

        # Allow JSON editing
        with st.container(border=True):
            json_text = st_monaco(
                value=json.dumps(json_data, indent=4),
                height="280px",
                language="json",
                theme="vs-dark",
            )

        # If "Save" is clicked, update the JSON file
        if st.button("💾 Save JSON Prompt"):
            with open(source_path, "w") as file:
                json.dump(json.loads(json_text), file, indent=4)
            st.toast("JSON Prompt successfully saved!")


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

    # Display view
    display_edit_json_sources(PROMPT_ROOT_DIR)


if __name__ == "__main__":
    main()
