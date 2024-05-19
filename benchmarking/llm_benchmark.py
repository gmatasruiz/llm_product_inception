# --- Imports ---
import os
import json
from benchmarking.classes.SpecificLLMBenchmark import *
from benchmarking.classes.BaseLLMBenchmark import BaseLLMBenchmark
import pandas as pd


# --- Functions ---
def get_benchmark_instance(model: str) -> BaseLLMBenchmark:
    """
    Return an instance of a specific benchmark class based on the provided model name.
    """
    benchmarks = {
        "Mixtral-8x7B": Mixtral8x7BBenchmark,
        "Meta-Llama3-8B": LlamaV38BBenchmark,
        "ChatGPT": ChatGPTBenchmark
    }
    if model in benchmarks:
        return benchmarks[model]()
    else:
        raise ValueError(f"No benchmark instance available for model: {model}")


def load_json_file(filepath: str) -> dict:
    """
    Load and return the content of a JSON file.
    """
    with open(filepath, 'r') as file:
        return json.load(file)


def read_input_data(base_dir: str, step_dir: str) -> tuple:
    """
    Read input data from the specified directories.
    """
    input_dir = os.path.join(base_dir, "input", step_dir)
    
    expected_response_path = os.path.join(input_dir, "expected_response", f"expected_response_{step_dir}.json")
    expected_response_data = load_json_file(expected_response_path)["data"]

    source_path = os.path.join(input_dir, "source", f"source_{step_dir}.json")
    source_data = load_json_file(source_path)["data"]
    
    template_dir = os.path.join(input_dir, "templates")

    return expected_response_data, source_data, template_dir


def create_prompt(template_data: str, source_data: str) -> str:
    """
    Create a prompt by replacing placeholders in the template with source data.
    """
    return template_data.replace("%%source", source_data)


def read_generated_response(output_filepath: str) -> str:
    """
    Read the generated response from the specified file.
    """
    return load_json_file(output_filepath)["data"]


def evaluate_responses(benchmark: BaseLLMBenchmark, expected_response: str, source_data: str, template_dir: str, output_path: str) -> list:
    """
    Evaluate responses for all templates in the template directory.
    """
    all_results = []

    for template_file in os.listdir(template_dir):
        template_path = os.path.join(template_dir, template_file)
        template_data = load_json_file(template_path)["data"]
        prompt = create_prompt(template_data, source_data)

        output_filename = template_file.replace("template", "llm_response")
        output_filepath = os.path.join(output_path, output_filename)
        llm_response = read_generated_response(output_filepath)

        results = benchmark.evaluate_response(prompt, expected_response, llm_response)
        all_results.append(results)
        print(f"Results for {template_file}: {results}")

    return all_results


def save_benchmark_results(benchmark: BaseLLMBenchmark, metrics_path: str, figures_path: str, results: list) -> None:
    """
    Save the benchmark results to files.
    """
    metrics_df = pd.DataFrame(results)
    benchmark.save_metrics_to_file(metrics_df, metrics_path, figures_path)


def benchmark_model_step(base_dir: str, model: str, step_dir: str) -> None:
    """
    Perform benchmarking for a specific model at a given step.
    """
    expected_response, source_data, template_dir = read_input_data(base_dir, step_dir)

    output_dir = os.path.join(base_dir, "results", model, step_dir, "output")
    metrics_path = os.path.join(base_dir, "results", model, step_dir, "metrics")
    figures_path = os.path.join(base_dir, "results", model, step_dir, "figures")

    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    benchmark = get_benchmark_instance(model)
    results = evaluate_responses(benchmark, expected_response, source_data, template_dir, output_dir)
    save_benchmark_results(benchmark, metrics_path, figures_path, results)


if __name__ == "__main__":
    """
    Main function to benchmark responses for each model and step, and save the results.
    """
    models = ["Mixtral-8x7B", "Meta-Llama3-8B", "ChatGPT"]
    root_dir = os.path.join(os.getcwd(), "prompt_creation", "prompting")

    input_path = os.path.join(root_dir, "input")
    if os.path.exists(input_path):
        steps = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
        for model in models:
            for step in steps:
                benchmark_model_step(root_dir, model, step)
