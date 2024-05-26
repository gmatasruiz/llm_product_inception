# Thesis on LLMs for Product Conception Efficiency - Implementation

## Overview

This documentation provides comprehensive information on the developed code for managing the pipelining, prompting, and benchmarking of Large Language Models (LLMs). The code includes functionality for handling different LLM instances, reading and processing prompts, generating responses, and evaluating the performance of responses using various models.

## Table of Contents

1. Introduction
2. Architecture
3. Functionality
   - LLM Instance Management
   - Prompt Processing
   - Metadata Creation
   - Model Step Processing
4. Benchmarking
   - Evaluation Metrics
   - Evaluation Process
5. Usage
   - Setting Up
   - Running the Pipeline
6. Class Descriptions
   - BaseLLMInstance
   - Specific LLM Instances
   - BaseLLMBenchmark
   - Specific LLM Benchmarks
7. Error Handling
8. Extending the Code
9. References

## 1. Introduction

This documentation covers the implementation and usage of a codebase designed to streamline the process of working with various LLMs. The primary goal is to provide a flexible and efficient way to handle prompts, generate responses, and evaluate the performance of different models.

## 2. Architecture

The codebase is structured into several key components:

- **LLM Instance Management:** Handles the creation and initialization of different LLM instances.
- **Prompt Processing:** Manages the reading, merging, and processing of prompts.
- **Metadata Creation:** Constructs metadata for each prompt-response pair.
- **Model Step Processing:** Oversees the sequential processing of templates for each model and step.
- **Benchmarking:** Evaluates the performance of LLM responses based on various metrics.

## 3. Functionality

### LLM Instance Management

- **get_llm_instance(model: str):** Returns an instance of a specific LLM based on the provided model name. Supported models include "Mixtral-8x7B", "Meta-Llama3-8B", and "ChatGPT".

### Prompt Processing

- **process_single_template(instance, source_path, template_path, output_path):** Reads the source and template files, processes the prompt using the specified LLM instance, and writes the response to the output directory.

### Metadata Creation

- **create_meta(source_path, template_path, instance):** Constructs a metadata dictionary that includes information such as file paths, model used, step number, and timestamp.

### Model Step Processing

- **process_model_step(base_dir, model, step_dir):** Processes a specific model step by iterating through templates, generating responses, and saving them to the output directory.

## 4. Benchmarking

### Evaluation Metrics

The benchmarking process includes various metrics to evaluate the performance of LLM responses:

- **Relevance (Semantic Similarity):** Measures how semantically similar the LLM response is to the expected response.
- **Accuracy:** Assesses the cosine similarity between the vectorized expected response and the LLM response.
- **Coherence:** Evaluates the logical flow and structure of the response.
- **Creativity:** Measures the lexical diversity of the response that is not present in the prompt.
- **Engagement:** Assesses the emotional engagement and structure of the response.
- **Sentiment Alignment:** Evaluates the alignment of sentiment between the expected response and the LLM response.

### Evaluation Process

The benchmarking process involves the following steps:

1. **Preprocessing:** Tokenizes the input text, removes stopwords, and converts the tokens to lowercase.
2. **Vectorization:** Transforms text into numerical vectors using techniques such as TF-IDF and BERT.
3. **Dimensionality Reduction:** Reduces the dimensionality of vectors for visualization using PCA.
4. **Semantic Similarity Calculation:** Uses Word2Vec to calculate the semantic similarity between responses.
5. **Metric Evaluation:** Computes various evaluation metrics for LLM responses.
6. **Results Visualization:** Plots the evaluation metrics using Plotly Express.

## 5. Usage

### Setting Up

1. Ensure all dependencies are installed.
2. Configure environment variables, including tokens for accessing models.
3. Organize the directory structure with input and output folders as described.

### Running the Pipeline

1. Define the list of models to use and the root directory for processing.
2. Execute the main script, which will iterate through the models and steps, generating responses for each template.

## 6. Class Descriptions

### BaseLLMInstance

- **Overview:** An abstract base class providing a common interface for LLM instances.
- **Attributes:**
  - `model_name`: Name of the LLM.
  - `model_id`: Identifier for the LLM.
  - `device`: Device on which the model is initialized.
  - `hf_token`: Hugging Face token.
  - `model`: The language model.
  - `tokenizer`: The tokenizer for the language model.
- **Methods:**
  - `__init__(self, model_name, model_id)`: Initializes the LLM instance.
  - `get_model_device(self, device_map)`: Returns the device for model initialization.
  - `init_model(self, model_kwargs)`: Initializes the model and tokenizer.
  - `read_prompt(self, source_file_path, template_file_path)`: Reads and merges source and template data.
  - `process_prompt(self, prompt, **llm_kwargs)`: Abstract method for processing the prompt.
  - `write_response(self, filepath, response, meta)`: Writes the response to a file.

### Specific LLM Instances

- **ChatGPTInstance:** Manages the ChatGPT model.
- **Mixtral8x7BInstance:** Manages the Mixtral 8x7B model.
- **LlamaV38BInstance:** Manages the Llama V3 8B model.

Each class inherits from `BaseLLMInstance` and implements the abstract methods for initializing the model, processing prompts, and writing responses.

### BaseLLMBenchmark

- **Overview:** A base class for evaluating the performance of LLMs.
- **Attributes:**
  - `vectorizer`: The TF-IDF vectorizer used for vectorizing prompts.
  - `stop_words`: A set of stopwords for text preprocessing.
  - `metrics_history`: A list to store the evaluation results of LLM responses.
  - `prompt_vectors`: A list to store the vectorized representations of prompts.
- **Methods:**
  - `preprocess(text)`: Tokenizes the input text, removes stopwords, and converts the tokens to lowercase.
  - `vectorize_prompt(prompt)`: Vectorizes the prompt using TF-IDF.
  - `plot_prompt_centroids(n_components)`: Plots the centroids of the vectorized prompts.
  - `semantic_similarity(model, tokens1, tokens2)`: Calculates the average similarity between two sets of tokens using Word2Vec.
  - `bert_vectorize(text)`: Vectorizes the input text using BERT.
  - `evaluate_accuracy(expected_response, llm_response)`: Calculates the accuracy score between the expected response and the LLM response.
  - `evaluate_coherence(response)`: Calculates the coherence score of a response.
  - `evaluate_creativity(prompt_tokens, response_tokens)`: Calculates the creativity score of a response.
  - `evaluate_engagement(response)`: Calculates the engagement score of a response.
  - `evaluate_sentiment_alignment(expected_response, llm_response)`: Evaluates the sentiment alignment between the expected response and the LLM response.
  - `evaluate_response(prompt, expected_response, llm_response)`: Evaluates the response generated by the LLM based on multiple metrics.
  - `write_metrics_df(write_to_json_file, path)`: Writes the evaluation metrics history to a DataFrame and optionally to a JSON file.
  - `write_metrics_fig()`: Generates a bar plot visualizing the benchmarking metrics.
  - `save_metrics_to_file(output_fname, metrics_path, figures_path)`: Saves the evaluation metrics to files.
  - `plot_metrics()`: Plots all benchmarking metrics based on the history of evaluations.

### Specific LLM Benchmarks

- **ChatGPTBenchmark:** Evaluates the performance of ChatGPT.
- **Mixtral8x7BBenchmark:** Evaluates the performance of Mixtral 8x7B.
- **LlamaV38BBenchmark:** Evaluates the performance of Llama V3 8B.

Each class inherits from `BaseLLMBenchmark` and implements the methods for evaluating the performance of the respective LLM.

## 7. Error Handling

- **ValueError:** Raised when an unsupported model name is provided.
- **FileNotFoundError:** Ensures that source and template files are accessible before processing.
- **Exception Handling:** General error handling to manage unexpected issues during processing.

## 8. Extending the Code

To extend the code to support additional models:

1. Create a new class inheriting from `BaseLLMInstance`.
2. Implement the abstract methods specific to the new model.
3. Update the `get_llm_instance` function to include the new model.

## 9. References

- [Hugging Face Documentation](https://huggingface.co/docs)
- [OpenAI API Documentation](https://beta.openai.com/docs)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MLX Software Suite Documentation](https://github.com/ml-explore)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) by Mikolov et al.
- [A Method for the Construction of Minimum-Redundancy Codes](https://ieeexplore.ieee.org/document/1056774) by David A.
