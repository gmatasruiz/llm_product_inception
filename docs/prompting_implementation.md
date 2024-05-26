
# Prompt Strategies for AI-Powered Product Development

## Overview

This document provides comprehensive documentation for the developed code that manages the pipelining and prompting of Large Language Models (LLMs). The code includes functionality for handling different LLM instances, reading and processing prompts, and generating responses using various models. 

## Table of Contents

1. Introduction
2. Architecture
3. Functionality
   - LLM Instance Management
   - Prompt Processing
   - Metadata Creation
   - Model Step Processing
4. Usage
   - Setting Up
   - Running the Pipeline
5. Class Descriptions
   - BaseLLMInstance
   - Specific LLM Instances
6. Error Handling
7. Extending the Code
8. References

## 1. Introduction

This documentation covers the implementation and usage of a codebase designed to streamline the process of working with various LLMs. The primary goal is to provide a flexible and efficient way to handle prompts, generate responses, and manage different models.

## 2. Architecture

The codebase is structured into several key components:

- **LLM Instance Management:** Handles the creation and initialization of different LLM instances.
- **Prompt Processing:** Manages the reading, merging, and processing of prompts.
- **Metadata Creation:** Constructs metadata for each prompt-response pair.
- **Model Step Processing:** Oversees the sequential processing of templates for each model and step.

## 3. Functionality

### LLM Instance Management

- **get_llm_instance(model: str):** Returns an instance of a specific LLM based on the provided model name. Supported models include "Mixtral-8x7B", "Meta-Llama3-8B", and "ChatGPT".

### Prompt Processing

- **process_single_template(instance, source_path, template_path, output_path):** Reads the source and template files, processes the prompt using the specified LLM instance, and writes the response to the output directory.

### Metadata Creation

- **create_meta(source_path, template_path, instance):** Constructs a metadata dictionary that includes information such as file paths, model used, step number, and timestamp.

### Model Step Processing

- **process_model_step(base_dir, model, step_dir):** Processes a specific model step by iterating through templates, generating responses, and saving them to the output directory.

## 4. Usage

### Setting Up

1. Ensure all dependencies are installed.
2. Configure environment variables, including tokens for accessing models.
3. Organize the directory structure with input and output folders as described.

### Running the Pipeline

1. Define the list of models to use and the root directory for processing.
2. Execute the main script, which will iterate through the models and steps, generating responses for each template.

## 5. Class Descriptions

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

## 6. Error Handling

- **ValueError:** Raised when an unsupported model name is provided.
- **FileNotFoundError:** Ensures that source and template files are accessible before processing.
- **Exception Handling:** General error handling to manage unexpected issues during processing.

## 7. Extending the Code

To extend the code to support additional models:
1. Create a new class inheriting from `BaseLLMInstance`.
2. Implement the abstract methods specific to the new model.
3. Update the `get_llm_instance` function to include the new model.

## 8. References

- [Hugging Face Documentation](https://huggingface.co/docs)
- [OpenAI API Documentation](https://beta.openai.com/docs)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MLX Software Suite Documentation](https://github.com/ml-explore)
