# --- Imports ---
import streamlit as st

from utils.utils import *
from prompt_creation.local_llm_prompting import *


# --- Functions ---
@st.experimental_dialog("LLM Prompting", width="large")
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
        # Assert at least one step has been selected
        assert len(steps) > 0

        # Initialize progress bar
        status = st.status(
            label="Generating LLM responses...", state="running", expanded=False
        )
        progress_bar = st.progress(0)

        # Define for loop for operations
        total_iter = len(models) * len(steps)
        for model in models:
            i = models.index(model) + 1
            for step in steps:  # Assuming steps 1, 2, 3; adjust range as needed
                j = steps.index(step) + 1
                step_dir = f"step{step}"

                # Run the prompting process
                process_model_step(root_dir, model, step_dir)

                # Update display elements (progress)
                status.update(
                    label="Generating LLM responses...", state="running", expanded=True
                )

                progress_bar.progress(value=i * j / total_iter)

        # When finished, remove progress elements
        progress_bar.empty()
        status.update(
            label=f"Prompting finished successfully for steps: [{','.join(steps)}]",
            state="complete",
            expanded=False,
        )

    except Exception as e:
        status.update(label="Error", state="error", expanded=True)

        with status:
            st.exception(e)


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
            selected_llms = comp_select_model(
                mode="multiple",
                sidebar=False,
                multi_checkbox=True,
            )

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
    )


def comp_select_step(
    mode: str = "single",
    sidebar: bool = True,
    multi_checkbox: bool = False,
    num_only: bool = True,
):
    """
    This function displays a selectbox or checkboxes to allow the user to choose one or multiple steps from a given list of available steps.

    Parameters:
        mode (str, optional): The mode of selection. Can be "single" for selecting one step or "multiple" for selecting multiple steps. Defaults to "single".
        sidebar (bool, optional): Determines whether the selectbox or checkboxes are displayed in the sidebar. Defaults to True.
        multi_checkbox (bool, optional): Determines whether to use checkboxes for multiple selection. If False, a multiselect dropdown will be used. Defaults to False.
        numonly (bool, optional): Determines whether to return the whole string or only the numeric part.

    Returns:
        list: A list of selected steps.

    Example:
        selected_steps = comp_select_step(mode="single", sidebar=True, multi_checkbox=False)
        selected_steps = comp_select_step(mode="multiple", sidebar=False, multi_checkbox=True)
    """
    steps = comp_select_entity(
        entity_list=AVAILABLE_STEPS,
        label="Step:",
        format_func=get_step_n,
        mode=mode,
        sidebar=sidebar,
        multi_checkbox=multi_checkbox,
    )

    if num_only:
        steps = list(map(get_step_n, steps))

    return steps


def comp_select_model(
    mode: str = "multiple", sidebar: bool = False, multi_checkbox: bool = True
):
    """
    This function displays a selectbox or checkboxes to allow the user to choose one or multiple models from a given list of available models.

    Parameters:
        mode (str, optional): The mode of selection. Can be "single" for selecting one model or "multiple" for selecting multiple models. Defaults to "multiple".
        sidebar (bool, optional): Determines whether the selectbox or checkboxes are displayed in the sidebar. Defaults to False.
        multi_checkbox (bool, optional): Determines whether to use checkboxes for multiple selection. If False, a multiselect dropdown will be used. Defaults to True.

    Returns:
        list: A list of selected models.

    Example:
        selected_models = comp_select_model(mode="single", sidebar=True, multi_checkbox=False)
        selected_models = comp_select_model(mode="multiple", sidebar=False, multi_checkbox=True)
    """
    return comp_select_entity(
        entity_list=LLMs,
        label="Model:",
        format_func=lambda x: x.replace("-", " "),
        mode=mode,
        sidebar=sidebar,
        multi_checkbox=multi_checkbox,
    )


def comp_select_entity(
    entity_list: list,
    label: str = "",
    format_func=None,
    mode: str = "single",
    sidebar: bool = False,
    multi_checkbox: bool = False,
):
    """
    This function displays a selectbox or checkboxes to allow the user to choose one or multiple entities from a given list.

    Parameters:
        entity_list (list): A list of entities to choose from.
        label (str, optional): The label to display above the selectbox or checkboxes. Defaults to an empty string.
        format_func (function, optional): A function to format the display of each entity in the selectbox or checkboxes. Defaults to None.
        mode (str, optional): The mode of selection. Can be "single" for selecting one entity or "multiple" for selecting multiple entities. Defaults to "single".
        sidebar (bool, optional): Determines whether the selectbox or checkboxes are displayed in the sidebar. Defaults to False.
        multi_checkbox (bool, optional): Determines whether to use checkboxes for multiple selection. If False, a multiselect dropdown will be used. Defaults to False.

    Returns:
        list: A list of selected entities.

    Raises:
        NotImplementedError: If an unsupported mode is specified.

    Example:
        selected_entities = comp_select_entity(entity_list=["entity1", "entity2", "entity3"], label="Select Entity", mode="single", sidebar=True)
        selected_entities = comp_select_entity(entity_list=["entity1", "entity2", "entity3"], label="Select Entities", mode="multiple", sidebar=False, multi_checkbox=True)
    """

    base_comp = st.sidebar.empty() if sidebar else st.empty()

    if mode == "single":
        selected_entities = base_comp.selectbox(
            options=entity_list, label=label, format_func=format_func
        )
        selected_entities = [
            selected_entities,
        ]

    elif mode == "multiple":
        with base_comp.container():

            if multi_checkbox:
                selected_entities = [
                    entity for entity in entity_list if st.checkbox(entity, value=False)
                ]
            else:
                selected_entities = st.multiselect(
                    options=entity_list, label=label, format_func=format_func
                )

    else:
        NotImplementedError

    return selected_entities


def comp_display_std_output(mode: str = "stdout"):
    """
    Display standard output or standard error based on the specified mode.

    Parameters:
        mode (str, optional): The mode of output to display. Can be "stdout" for standard output or "stderr" for standard error. Defaults to "stdout".

    Raises:
        NotImplementedError: If an unsupported mode is specified.

    Returns:
        None

    """
    # Set header to show which terminal output is being shown
    st.header(body=mode)

    # Retrieve terminal output
    stdout, stderr = retrieve_terminal_output()

    # Display the code depending on the mode
    if mode == "stdout" and stdout:
        st.code(stdout)
    elif mode == "stderr" and stdout:
        st.code(stderr)
    else:
        raise NotImplementedError
