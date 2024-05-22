# --- Imports ---
import streamlit as st

from utils.utils import *
from utils.streamlit_ui.ui_components import (
    comp_select_step,
    comp_select_model,
    comp_batch_process_options,
    comp_display_chart_from_file,
    comp_display_table_from_file,
    comp_display_text_alongside,
    st_markdown_spacer,
)


# --- Functions ---


def display_step_response_comparative(
    selected_source: str,
    selected_expected_response: str,
    templates_dir: str,
    llm_response_dir: str,
):
    """
    Display the prompt, LLM response, and expected response for a selected source.

    Parameters:
        - selected_source (str): The path to the selected source file.
        - selected_expected_response (str): The path to the selected expected response file.
        - templates_dir (str): The directory containing templates.
        - llm_response_dir (str): The directory containing LLM responses.

    Returns:
        None
    """

    # Display prompt
    with st.columns(2)[0]:
        # Show prompt and LLM response
        ## Extract files from selection
        selected_template_fname = st.selectbox(
            "Template:", sorted(os.listdir(templates_dir))
        )

    try:
        selected_template = os.path.join(templates_dir, selected_template_fname)

        selected_llm_response = os.path.join(
            llm_response_dir,
            selected_template_fname.replace("template", "llm_response"),
        )

        ## Extract data from selected files
        prompt = create_prompt(
            *[
                read_generated_response(fpath)
                for fpath in (selected_template, selected_source)
            ]
        )
        llm_response = read_generated_response(selected_llm_response)
        expected_response = read_generated_response(selected_expected_response)

        ## Display text
        comp_display_text_alongside(
            text_array=(prompt, llm_response, expected_response),
            label_array=("Prompt:", "LLM Response:", "Expected Response:"),
            elem_per_col=2,
            height=500,
        )

    except:
        st.error(
            "No valid data could be found. Please, run the benchmark process for the selected model and step."
        )


def display_step_results(root_dir: str):
    """
    Display the results of a selected step and model by drilling down into the data.

    Parameters:
        root_dir (str): The root directory where the data is stored.

    Returns:
        None
    """

    # Mandatory select step to drilldown
    selected_step = comp_select_step(
        mode="single", sidebar=True, multi_checkbox=False, num_only=True
    )
    str_selected_step = f"step{''.join(selected_step)}"

    # Mandatory select model to drilldown
    selected_model = comp_select_model(
        mode="single", sidebar=True, multi_checkbox=False
    )
    str_selected_model = f"{''.join(selected_model)}"

    # Add batch process component
    comp_batch_process_options(
        root_dir=root_dir,
        mode="steps",
        steps=selected_step,
        on_sidebar=True,
        spacer_n=1,
    )

    # Main view

    ## Filesystem dirs init
    ### Input
    source_dir = os.path.join(root_dir, "input", str_selected_step, "source")
    selected_source = os.path.join(source_dir, f"source_{str_selected_step}.json")

    templates_dir = os.path.join(root_dir, "input", str_selected_step, "templates")

    expected_response_dir = os.path.join(
        root_dir, "input", str_selected_step, "expected_response"
    )
    selected_expected_response = os.path.join(
        expected_response_dir, f"expected_response_{str_selected_step}.json"
    )

    ### Ouput
    figure_dir = os.path.join(
        root_dir, "results", str_selected_model, str_selected_step, "figures"
    )

    metrics_dir = os.path.join(
        root_dir, "results", str_selected_model, str_selected_step, "metrics"
    )

    llm_response_dir = os.path.join(
        root_dir, "results", str_selected_model, str_selected_step, "output"
    )

    # Display components
    if selected_step and selected_model:
        st.title(f"🔎 {str_selected_step.capitalize()} Drilldown")
        st.divider()

        # Display results figure
        with st.expander("Benchmark Results", expanded=True):
            st_markdown_spacer()
            tabs = st.tabs(("📈 Chart", "🗃 Table"))
            with tabs[0]:
                fig_is_available = comp_display_chart_from_file(
                    filepath=os.path.join(
                        figure_dir, f"metrics_{str_selected_step}.json"
                    )
                )

            with tabs[1]:
                data_is_available = comp_display_table_from_file(
                    filepath=os.path.join(
                        metrics_dir, f"metrics_{str_selected_step}.csv"
                    ),
                    ignore_index=True,
                    beautify_col_names=True,
                )

        if fig_is_available and data_is_available:
            with st.expander("Comparative", expanded=True):
                st_markdown_spacer()
                display_step_response_comparative(
                    selected_source,
                    selected_expected_response,
                    templates_dir,
                    llm_response_dir,
                )


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
    display_step_results(PROMPT_ROOT_DIR)


if __name__ == "__main__":
    main()
