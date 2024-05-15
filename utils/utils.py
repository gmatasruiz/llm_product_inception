# --- Imports ---
import os
import subprocess

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


def retrieve_terminal_output() -> str:
    """
    Retrieve the terminal output from running a bash command.

    This function executes a bash command using the 'subprocess' module and captures the stdout and stderr outputs. It then processes the outputs to remove any leading or trailing newlines and returns them as a tuple.

    Returns:
        str: The stdout output of the bash command.
        str: The stderr output of the bash command.

    Notes:
        - Subprocess relies on paralelism which can be extremely buggy when using the `transformers` module.

    Example:
        stdout, stderr = retrieve_terminal_output()
    """
    result = subprocess.Popen(
        ["bash", "run.sh"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    std_outputs = stdout, stderr = result.communicate()

    stdout_str, stderr_str = [
        "\n".join(std_output.decode().split("\n")[1:][:-1])
        for std_output in std_outputs
    ]

    return (stdout_str, stderr_str)


# --- Constants ---

REPO_ROOT_DIR = os.path.join(os.getcwd())
PROMPT_ROOT_DIR = os.path.join(os.getcwd(), "prompt_creation", "prompting")
AVAILABLE_STEPS = get_available_steps(PROMPT_ROOT_DIR)
AVAILABLE_STEPS_N = list(map(get_step_n, get_available_steps(PROMPT_ROOT_DIR)))
LLMs = ["Mixtral-8x7B", "Meta-Llama3-8B", "ChatGPT"]
