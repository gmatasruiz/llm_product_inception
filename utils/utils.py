# --- Imports ---
import os

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


# --- Constants ---

REPO_ROOT_DIR = os.path.join(os.getcwd())
PROMPT_ROOT_DIR = os.path.join(os.getcwd(), "prompt_creation", "prompting")
AVAILABLE_STEPS = get_available_steps(PROMPT_ROOT_DIR)
AVAILABLE_STEPS_N = list(map(get_step_n, get_available_steps(PROMPT_ROOT_DIR)))
LLMs = ["Mixtral-8x7B", "Meta-Llama3-8B", "ChatGPT"]
