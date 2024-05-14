# --- Imports ---
import os
import json
import datetime
from prompt_creation.classes.SpecificLLMInstance import *
from prompt_creation.classes.BaseLLMInstance import BaseLLMInstance


# --- Functions ---
def get_llm_instance(model: str):
    """
    Returns an instance of a specific language model based on the given model name.

    Parameters:
        model (str): The name of the language model.

    Returns:
        BaseLLMInstance: An instance of the specific language model.

    Raises:
        ValueError: If the given model name is not supported.

    Notes:
        - The supported model names are "Mixtral-8x7B", "Meta-Llama3-8B", and "ChatGPT".
        - The returned instance is an object of a class that extends the BaseLLMInstance abstract base class.
        - If the given model name is not supported, a ValueError is raised.
    """
    if model == "Mixtral-8x7B":
        return Mixtral8x7BInstance()
    elif model == "Meta-Llama3-8B":
        return LlamaV38BInstance()
    elif model == "ChatGPT":
        return ChatGPTInstance()
    else:
        raise ValueError(f"No LLM instance available for model: {model}")


def create_meta(
    source_path: str,
    template_path: str,
    instance: BaseLLMInstance,
):
    """
    Constructs the metadata dictionary from a given source file, template file, and instance.

    Parameters:
        source_path (str): The path to the source file.
        template_path (str): The path to the template file.
        instance (BaseLLMInstance): An instance of the specific language model.

    Returns:
        dict: The metadata dictionary containing the following keys:
            - "source_filepath": The filepath of the source file.
            - "template_filepath": The filepath of the template file.
            - "output_generated_on": The timestamp when the output was generated.
            - "model_used": The name of the language model used.
            - "step_number": The step number of the template.
            - "iteration_number": The iteration number of the template.

    """
    with open(source_path, "r") as file:
        source_data = json.load(file)
    with open(template_path, "r") as file:
        template_data = json.load(file)

    # Construct the metadata dictionary
    meta = {
        "source_filepath": source_data["__meta__"]["source_filepath"],
        "template_filepath": template_data["__meta__"]["template_filepath"],
        "output_generated_on": datetime.datetime.now().isoformat(),
        "model_used": instance.model_name,
        "step_number": template_data["__meta__"]["step_number"],
        "iteration_number": template_data["__meta__"]["iteration_number"],
    }
    return meta


def process_single_template(
    instance: BaseLLMInstance, source_path: str, template_path: str, output_path: str
):
    """
    This function processes a single template using an instance of the BaseLLMInstance class.

    Parameters:
        instance (BaseLLMInstance): An instance of the BaseLLMInstance class.
        source_path (str): The path to the source file.
        template_path (str): The path to the template file.
        output_path (str): The path where the output file will be saved.

    Returns:
        None

    Raises:
        None
    """
    prompt = instance.read_prompt(source_path, template_path)
    response = instance.process_prompt(prompt)
    output_filename = os.path.basename(template_path).replace("template", "response")

    meta = create_meta(source_path, template_path, instance)

    instance.write_response(os.path.join(output_path, output_filename), response, meta)


def process_model_step(base_dir: str, model: str, step_dir: str):
    """
    Process a model step.

    This function processes a model step by reading the source file, iterating through the templates in the template directory, and generating output files for each template.

    Parameters:
        base_dir (str): The base directory where the model step is located.
        model (str): The name of the language model.
        step (str): The name of the step.

    Returns:
        None

    Raises:
        None
    """
    input_dir = "input"
    output_dir = "results"

    # Input dirs
    source_path = os.path.join(base_dir, input_dir, step_dir, "source", f"source_{step_dir}.json")
    template_dir = os.path.join(base_dir, input_dir, step_dir, "templates" )
    expected_response_dir = os.path.join(base_dir, input_dir, step_dir, "expected_response")
    
    # Outpur dirs
    output_dir = os.path.join(base_dir, output_dir, model, step_dir, "output")
    
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    instance = get_llm_instance(model)

    for template_file in os.listdir(template_dir):
        template_path = os.path.join(template_dir, template_file)
        process_single_template(instance, source_path, template_path, output_dir)


if __name__ == "__main__":
    """
    This code snippet iterates through the templates in the template directory, reads the source file, and 
    generates output files for each template.


    """
    
    # Inputs
    """
    Inputs

    - models: A list of model names.
    - root_dir: The root directory where the model steps are located.
    """
    # models = ["Mixtral-8x7B", "Meta-Llama3-8B"]
    models = ["Meta-Llama3-8B"]
    root_dir = os.path.join(os.getcwd(), "prompt_creation", "prompting")

    # Main Loop
    """
    Flow

    1. Iterate through each model in the models list.
    2. Create the model path by joining the root_dir and the current model name.
    3. Check if the model path exists.
    4. Get a list of directories in the model path.
    5. Iterate through each step in the list of directories.
    6. Call the process_model_step function with the root_dir, current model name, and current step as arguments.
    """
        # - Overview: Run all prompts and benchmarks for any selected models -> Results overview
        # - Step_N: Run models and check benchmarks for any given step and models -> Detailed results per step plus prompt & response comparison
    for model in models:
        input_path = os.path.join(root_dir, "input")
        if os.path.exists(input_path):
            steps = [
                d
                for d in os.listdir(input_path)
                if os.path.isdir(os.path.join(input_path, d))
            ]
            for step in steps:
                process_model_step(root_dir, model, step)
