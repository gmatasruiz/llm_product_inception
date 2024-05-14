# --- Imports ---
import os
import json
import streamlit as st
from streamlit_monaco import st_monaco
from prompt_creation.local_llm_prompting import *

# --- Lambda Functions ---
get_step_n = lambda x: x.split("_")[-1] if isinstance(x, str) else x


# --- Functions ---
def get_available_steps(root_dir: str):
    # Define the directory containing the JSON templates
    input_dir = os.path.join(root_dir, "input")

    # List available steps
    steps = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ]

    return sorted(steps)


def run_batch_prompting(models: list, root_dir: str, steps: list[int]):
    for model in models:
        for step in steps:  # Assuming steps 1, 2, 3; adjust range as needed
            step_dir = f"step{step}"
            process_model_step(root_dir, model, step_dir)


def run_batch_benchmarking(models: list, root_dir: str):
    # Placeholder for actual benchmarking logic
    st.write(f"Running batch benchmarking for models: {models}")


def display_overview():
    st.header("Overview Results")
    # Placeholder for displaying overall benchmark and drilldowns
    st.write("Overall benchmarks and detailed drilldowns go here.")


def display_step_results(step):
    st.header(f"Step {step} Results")
    # Placeholder for displaying step-specific benchmarks and comparisons
    st.write(f"Benchmarks and comparisons for step {step} go here.")


def edit_json_templates(root_dir):
    st.title("Edit JSON Templates")

    # Define the directory containing the JSON templates
    input_dir = os.path.join(root_dir, "input")
    selected_step = None
    selected_template_file = None
    selected_step = None

    # Dropdown to select a step
    dropdown_columns = st.columns(2)
    with dropdown_columns[0]:
        selected_step = st.selectbox(
            "Select Step",
            get_available_steps(root_dir),
            format_func=get_step_n,
            index=None,
        )

    with dropdown_columns[1]:
        if selected_step:
            template_dir = os.path.join(input_dir, selected_step, "templates")
            template_files = [
                f for f in os.listdir(template_dir) if f.endswith(".json")
            ]

            # Dropdown to select a template file
            selected_template_file = st.selectbox(
                "Select Template File", template_files
            )

    if selected_template_file:
        template_path = os.path.join(template_dir, selected_template_file)

        # Load and display the selected JSON template
        with open(template_path, "r") as file:
            json_data = json.load(file)

        json_text = st_monaco(
            value=json.dumps(json_data, indent=4),
            height="400px",
            language="json",
            theme="vs-dark",
        )

        if st.button("Save JSON Template"):
            with open(template_path, "w") as file:
                json.dump(json.loads(json_text), file, indent=4)
            st.toast("JSON Template Saved")


# --- Main ---
# Streamlit App
def main():
    # App config
    st.set_page_config(layout="wide")

    st.sidebar.title("Navigation")
    view = st.sidebar.radio(
        "Select View", ("Overview", "Step Drilldown", "Edit JSON Templates")
    )

    st.sidebar.title("Models")
    llms = ["Mixtral-8x7B", "Meta-Llama3-8B", "ChatGPT"]
    selected_llms = [llm for llm in llms if st.sidebar.checkbox(llm, value=True)]

    root_dir = os.path.join(os.getcwd(), "prompt_creation", "prompting")

    if view == "Overview":
        st.sidebar.button(
            "Run Batch Prompting",
            on_click=run_batch_prompting,
            args=(selected_llms, root_dir),
        )
        st.sidebar.button(
            "Run Batch Benchmarking",
            on_click=run_batch_benchmarking,
            args=(selected_llms, root_dir),
        )
        display_overview()
    elif view == "Edit JSON Templates":
        edit_json_templates(root_dir)
    elif view == "Step Drilldown":
        st.sidebar.button(
            "Run Prompting",
            on_click=run_batch_prompting,
            args=(selected_llms, root_dir),
        )
        st.sidebar.button(
            "Run Benchmarking",
            on_click=run_batch_benchmarking,
            args=(selected_llms, root_dir),
        )
        selected_step = st.selectbox(
            "Step:",
            get_available_steps(root_dir),
            format_func=get_step_n,
            index=None,
        )
        display_step_results(get_step_n(selected_step))

    else:
        st.warning("Select other option")


if __name__ == "__main__":
    main()
