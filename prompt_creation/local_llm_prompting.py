# main_script.py located in prompt_creation directory

import os
import json
from classes.SpecificLLMInstance import Mixtral8x7BInstance, LlamaV38BInstance


def get_llm_instance(model):
    """Return an instance of the LLM based on the model name."""
    if model == "Mixtral-8x7B":
        return Mixtral8x7BInstance()
    elif model == "Meta-Llama3-8B":
        return LlamaV38BInstance()
    else:
        raise ValueError(f"No LLM instance available for model: {model}")


def process_single_template(instance, source_path, template_path, output_path):
    """Process a single template with the source data."""
    prompt = instance.read_prompt(source_path, template_path)
    response = instance.process_prompt(prompt)
    output_filename = os.path.basename(template_path).replace("template", "response")

    print(os.path.join(output_path, output_filename))

    instance.write_response(os.path.join(output_path, output_filename), response)


def process_model_step(base_dir, model, step):
    """Process all templates for a single model and step."""
    source_path = os.path.join(base_dir, model, step, "source", f"source_{step}.json")
    template_dir = os.path.join(base_dir, model, step, "templates")
    output_dir = os.path.join(base_dir, model, step, "outputs")
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    instance = get_llm_instance(model)

    for template_file in os.listdir(template_dir):
        template_path = os.path.join(template_dir, template_file)
        process_single_template(instance, source_path, template_path, output_dir)


if __name__ == "__main__":
    # models = ["Mixtral-8x7B", "Meta-Llama3-8B"]
    models = ["Meta-Llama3-8B"]
    root_dir = os.path.join(os.getcwd(), "prompt_creation", "results")
    for model in models:
        model_path = os.path.join(root_dir, model)
        if os.path.exists(model_path):
            steps = [
                d
                for d in os.listdir(model_path)
                if os.path.isdir(os.path.join(model_path, d))
            ]
            for step in steps:
                process_model_step(root_dir, model, step)
