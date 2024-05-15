# --- Imports ---
import os
import json
import streamlit as st
from streamlit_monaco import st_monaco
from prompt_creation.local_llm_prompting import *

# --- Lambda Functions ---
get_step_n = lambda x: x.split("step")[-1] if isinstance(x, str) else x


# --- Functions ---
def get_available_steps(root_dir: str):
    """
    Get the available steps from the specified root directory.

    This function retrieves the available steps by listing the directories in the 'input' directory
    located in the specified root directory. The steps are sorted in ascending order.

    Parameters:
        root_dir (str): The root directory.

    Returns:
        list: A sorted list of available steps.

    Example:
        get_available_steps("/path/to/root/dir")
    """
    # Define the directory containing the JSON templates
    input_dir = os.path.join(root_dir, "input")

    # List available steps
    steps = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ]

    return sorted(steps)


@st.experimental_dialog("LLM Prompting")
def run_batch_prompting(models: list, root_dir: str, steps: list[str]):
    """
    Run batch prompting for a list of models and steps, with progress updates.

    This function initiates a batch prompting process for each specified model and step.
    It updates a progress bar and status messages throughout the operation to provide
    feedback on the current state of the process.

    Parameters:
        models (list): A list of models to run prompting on.
        root_dir (str): The root directory.
        steps (list[str]): A list of steps to perform prompting on.

    Returns:
        None

    Raises:
        Exception: If an error occurs during prompting.

    Features:
        - Progress Bar: Displays the progress of the prompting process.
        - Status Messages: Provides real-time updates on the process state, including completion and error notifications.

    Example:
        run_batch_prompting(["model1", "model2"], "/path/to/root/dir", ["1", "2", "3"])
    """

    try:
        assert len(steps) > 0

        st.status("", state="running")
        progress_bar = st.progress(0, text="Generating LLM responses...")

        total_iter = len(models) * len(steps)

        for model in models:
            i = models.index(model) + 1
            for step in steps:  # Assuming steps 1, 2, 3; adjust range as needed
                j = steps.index(step) + 1
                step_dir = f"step{step}"
                process_model_step(root_dir, model, step_dir)
                progress_bar.progress(
                    value=i * j / total_iter, text="Generating LLM responses..."
                )

        # When finished...
        progress_bar.empty()
        st.status("Prompting finished!", state="complete")

    except Exception as e:
        st.status(f"{e}", state="error")


def run_batch_benchmarking(models: list, root_dir: str):
    """
    Run batch benchmarking for a list of models.

    Parameters:
        models (list): A list of models to run benchmarking on.
        root_dir (str): The root directory.

    Returns:
        None

    Raises:
        Exception: If an error occurs during benchmarking.

    Example:
        run_batch_benchmarking(["model1", "model2"], "/path/to/root/dir")
    """
    try:
        # Placeholder for actual benchmarking logic
        st.write(f"Running benchmarking for models: {models}")
        # TODO: Implement benchmark processing
        st.status("Benchmarking finished!", status="complete")
    except Exception as e:
        st.status(e, state="error")


def comp_batch_process_options(
    root_dir: str,
    mode: str = "steps",
    **kwargs,
):
    """
    This function generates a batch process options component for launching batch prompting or benchmarking.

    Parameters:
        root_dir (str): The root directory to retrieve the input files from.
        mode (str, optional): The mode of operation. Can be "batch" for batch processing or "steps" for step-wise processing. Defaults to "steps".
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    with st.columns(3)[-1]:

        with st.popover("Process Launcher"):
            selected_llms = [llm for llm in LLMs if st.checkbox(llm, value=False)]

            cols = st.columns(2)

            with cols[0]:
                prompt_button = comp_process_button(
                    root_dir=root_dir,
                    selected_llms=selected_llms,
                    process="prompt",
                    mode=mode,
                    **kwargs,
                )

            with cols[1]:
                benchmark_button = comp_process_button(
                    root_dir=root_dir,
                    selected_llms=selected_llms,
                    process="benchmark",
                    mode=mode,
                    **kwargs,
                )


def comp_process_button(
    root_dir: str,
    selected_llms: list[str],
    process: str = "prompt",
    mode: str = "steps",
    **kwargs,
):
    """
    This function creates a process button for performing batch prompting or benchmarking.

    Parameters:
        root_dir (str): The root directory.
        selected_llms (list[str]): A list of selected LLMs.
        process (str, optional): The process to perform. Can be "prompt" for batch prompting or "benchmark" for benchmarking. Defaults to "prompt".
        mode (str, optional): The mode of operation. Can be "batch" for batch processing or "steps" for step-wise processing. Defaults to "steps".
        **kwargs: Additional keyword arguments. Only use: "steps" = ["1","2","5"]

    Raises:
        NotImplementedError: If an unsupported process or mode is specified.

    Returns:
        None
    """
    # Initialize variables depending on mode
    ## Process
    if process == "prompt":
        callback_func = run_batch_prompting
    elif process == "benchmark":
        callback_func = run_batch_benchmarking
    else:
        raise NotImplementedError

    ## Mode
    if mode == "batch":
        steps = AVAILABLE_STEPS_N
    elif mode == "steps":
        steps = kwargs.get("steps", [])
    else:
        raise NotImplementedError

    process_button = st.button(
        f"{mode.capitalize()} {process.capitalize()}",
        on_click=callback_func,
        args=(
            selected_llms,
            root_dir,
            steps,
        ),
        # key=f"pbutton_{process}_{mode}_{str(steps)}",
    )


def comp_select_step():
    """
    This function displays a selectbox in the sidebar to allow the user to choose a step from a list of available steps.

    Returns:
        str: The selected step.

    """
    with st.sidebar:
        selected_step = st.selectbox(
            "Step:",
            AVAILABLE_STEPS,
            format_func=get_step_n,
        )
    return selected_step


def display_overview():
    comp_batch_process_options(root_dir=ROOT_DIR, mode="batch")

    st.title("Overview Results")
    # Placeholder for displaying overall benchmark and drilldowns
    # TODO: Implement benchmark visualization
    st.write("Overall benchmarks and detailed drilldowns go here.")


def display_step_results():
    """
    Display the results for a selected step.

    This function prompts the user to select a step from a list of available steps. It then calls
    the 'comp_batch_process_options' function to process the selected step. After processing, it
    displays the results for the selected step.

    Parameters:
        None

    Returns:
        None

    Example:
        display_step_results()
    """
    selected_step = comp_select_step()
    comp_batch_process_options(
        root_dir=ROOT_DIR,
        mode="steps",
        steps=[
            selected_step,
        ],
    )

    if selected_step:
        st.title(f"{selected_step.capitalize()} Results")

        # Placeholder for displaying step-specific benchmarks and comparisons
        # TODO: Implement benchmark visualization
        st.write(f"Benchmarks and comparisons for {selected_step} go here.")


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
    selected_step = comp_select_step()

    # Add batch process options
    comp_batch_process_options(
        root_dir=ROOT_DIR,
        mode="steps",
        steps=[
            selected_step,
        ],
    )

    # Radio group to select the file type to be modified
    st.title("Edit Sources")
    source_types = ["source", "templates", "expected_response"]
    source_type = st.radio(
        label="Select a source type:",
        options=source_types,
        format_func=lambda x: x.replace("_", " ").capitalize(),
        horizontal=True,
    )

    if selected_step and source_type in source_types:
        source_dir = os.path.join(input_dir, selected_step, source_type)
        source_files = [f for f in os.listdir(source_dir) if f.endswith(".json")]

        # Dropdown to select a template file
        selected_file = st.selectbox(f"Select {source_type} file:", source_files)

    if selected_file:
        source_path = os.path.join(source_dir, selected_file)

        # Load and display the selected JSON template
        with open(source_path, "r") as file:
            json_data = json.load(file)

        # Allow JSON editing
        json_text = st_monaco(
            value=json.dumps(json_data, indent=4),
            height="300px",
            language="json",
            theme="vs-dark",
        )

        # If "Save" is clicked, update the JSON file
        if st.button("💾 Save JSON Prompt"):
            with open(source_path, "w") as file:
                json.dump(json.loads(json_text), file, indent=4)
            st.toast("JSON Prompt successfully saved!")


# --- Constants ---

ROOT_DIR = os.path.join(os.getcwd(), "prompt_creation", "prompting")
AVAILABLE_STEPS = get_available_steps(ROOT_DIR)
AVAILABLE_STEPS_N = list(map(get_step_n, get_available_steps(ROOT_DIR)))
LLMs = ["Mixtral-8x7B", "Meta-Llama3-8B", "ChatGPT"]


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
    st.set_page_config(layout="wide")

    # Sidebar
    st.sidebar.title("Navigation")
    view = st.sidebar.radio(
        "Select View", ("Overview", "Step Drilldown", "Edit Prompts")
    )

    # Main views
    if view == "Overview":
        display_overview()
    elif view == "Edit Prompts":
        display_edit_json_sources(ROOT_DIR)
    elif view == "Step Drilldown":
        display_step_results()

    else:
        st.warning("Select other option")


if __name__ == "__main__":
    main()
