# --- Imports ---
import streamlit as st
import math

from utils.utils import *
from prompt_creation.llm_prompting import *
from benchmarking.llm_benchmark import *


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


@st.experimental_dialog("LLM Benchmarking", width="large")
def run_batch_benchmarking(models: list, root_dir: str, steps: list[str]):
    """
    Run batch prompbenchmarkingting for a list of models and steps, with progress updates.

    This function initiates a batch benchmarking process for each specified model and step.
    It updates a progress bar and status messages throughout the operation to provide
    feedback on the current state of the process.

    Parameters:
        models (list): A list of models to run benchmarking on.
        root_dir (str): The root directory.
        steps (list[str]): A list of steps to perform benchmarking on.

    Returns:
        None

    Raises:
        Exception: If an error occurs during benchmarking.

    Features:
        - Progress Bar: Displays the progress of the benchmarking process.
        - Status Messages: Provides real-time updates on the process state, including completion and error notifications.

    Example:
        run_batch_benchmarking(["model1", "model2"], "/path/to/root/dir", ["1", "2", "3"])
    """

    try:
        # Assert at least one step has been selected
        assert len(steps) > 0

        # Initialize progress bar
        status = st.status(
            label="Benchmarking LLM responses...", state="running", expanded=False
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
                benchmark_model_step(root_dir, model, step_dir)

                # Update display elements (progress)
                status.update(
                    label="Benchmarking LLM responses...",
                    state="running",
                    expanded=True,
                )

                progress_bar.progress(value=i * j / total_iter)

        # When finished, remove progress elements
        progress_bar.empty()
        status.update(
            label=f"Benchmarking finished successfully for steps: [{','.join(steps)}]",
            state="complete",
            expanded=False,
        )

    except Exception as e:
        status.update(label="Error", state="error", expanded=True)

        with status:
            st.exception(e)


def comp_batch_process_options(
    root_dir: str,
    mode: str = "steps",
    on_sidebar: bool = False,
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

        # Get values depending on `on_sidebar` argument
        if on_sidebar:
            sidebar_ = st.sidebar
            spacer_n = 1

        else:
            sidebar_ = st.empty()
            spacer_n = 0

        # Add spaces
        spacer_n = (
            kwargs.get("spacer_n", 0) if kwargs.get("spacer_n", 0) >= 0 else spacer_n
        )
        spaces = [st_markdown_spacer(on_sidebar=on_sidebar) for i in range(spacer_n)]

        # Using the popover
        with sidebar_.popover("Process Launcher"):
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


def comp_display_chart_from_file(filepath: str):
    """
    Display a Plotly chart from a JSON file.

    Parameters:
        filepath (str): The file path to the JSON file containing the Plotly chart data.

    Returns:
        True if successfull, otherwise False

    Raises:
        None
    """

    successful = False

    if os.path.exists(filepath) and filepath.endswith("json"):
        with open(filepath, "r") as file:
            fig_data = json.load(file)

        st.plotly_chart(figure_or_data=fig_data, use_container_width=True)

        successful = True

    else:
        st.warning(
            "No valid data could be found. Please, run the benchmark process for the selected model and step."
        )

    return successful


def comp_display_table_from_file(
    filepath: str,
    ignore_index: bool = False,
    index_column: str = None,
    beautify_col_names: bool = True,
):
    """
    Display a table from a CSV file.

    Parameters:
        filepath (str): The file path to the CSV file containing the table data.
        ignore_index (bool, optional): Whether to ignore the index column. Defaults to False.
        index_column (str, optional): The column to set as the index. Defaults to None.
        beautify_col_names (bool, optional): Whether to beautify column names by capitalizing and replacing underscores with spaces. Defaults to True.

    Returns:
        bool: True if the table is displayed successfully, otherwise False.

    Raises:
        None
    """

    successful = False
    st_column_config = {}

    # Load data from CSV file
    if os.path.exists(filepath) and filepath.endswith("csv"):
        df = pd.read_csv(filepath, index_col=None)

        # DF Configuration
        ## If index_column in column, set index
        if (not ignore_index) and (index_column in df.columns):
            df.set_index(index_column, inplace=True)

        ## If beautify_col_names, apply transformation to col_names
        st_column_config = {
            col: col.capitalize().replace("_", " ") for col in df.columns
        }

        st.dataframe(
            data=df,
            use_container_width=True,
            hide_index=ignore_index,
            column_config=st_column_config,
        )

        successful = True

    else:
        st.warning(
            "No valid data could be found. Please, run the benchmark process for the selected model and step."
        )

    return successful


def comp_display_text_alongside(
    text_array: list[str],
    label_array: list[str],
    elem_per_col: int = 1,
    height: int = 350,
):
    """
    Display text alongside labels in multiple columns.

    Parameters:
        text_array (list[str]): List of text values to display.
        label_array (list[str]): List of labels corresponding to the text values.
        elem_per_col (int, optional): Number of elements to display per column. Defaults to 1.
        height (int, optional): Height of each column. Defaults to 350.

    Returns:
        None

    Raises:
        st.error: If the number of labels does not match the number of text values.
    """
    if len(label_array) != len(text_array):
        st.error("Mismatch between the number of labels and values...")
        return

    n_cols = int(math.ceil(len(text_array) / elem_per_col))

    # Set Styling
    cols = [col.container(border=True, height=height) for col in st.columns(n_cols)]

    # Fill in information
    for idx, (text, label) in enumerate(zip(text_array, label_array)):
        col_idx = idx // elem_per_col
        with cols[col_idx]:
            st_markdown_color_text(f"**{label}**", "#68C3E0")
            st_markdown_color_text(f"""*"{text}"*""", "white")
            st_markdown_spacer()


def st_markdown_color_text(text: str, bgcolor: str = "white"):
    """
    Display colored text using Markdown syntax.

    Parameters:
        text (str): The text to be displayed.
        bgcolor (str, optional): The background color of the text. Defaults to "white".

    Returns:
        None
    """
    st.markdown(
        f'<span style="color:{bgcolor}"> {text}</span>',
        unsafe_allow_html=True,
    )


def st_markdown_spacer(on_sidebar: bool = False):
    """
    This function inserts a markdown spacer in the Streamlit app.

    Parameters:
        None

    Returns:
        None
    """

    if on_sidebar:
        spacer_ = st.sidebar.empty()

    else:
        spacer_ = st.empty()

    spacer_.markdown(
        f"<br>",
        unsafe_allow_html=True,
    )
