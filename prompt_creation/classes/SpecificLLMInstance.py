# --- Imports ---
import os

import torch
import transformers
import mlx_lm
from openai import OpenAI

from prompt_creation.classes.BaseLLMInstance import BaseLLMInstance


# --- Constants ---
MODEL_DEF = {
    "mixtral-8x7b": {
        "display_name": "Mixtral 8x7B",
        "name": "mixtral-8x-7b",
        "basemodel_hf_id": "mistralai/Mixtral-8x7B-v0.1",
        "mlx_hf_id": "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
    },
    "meta-llama-3-8b": {
        "display_name": "Meta Llama3 8B",
        "name": "Meta-Llama-3-8B",
        "basemodel_hf_id": "meta-llama/Meta-Llama-3-8B",
        "mlx_hf_id": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    },
}


# --- Classes ---
class ChatGPTInstance(BaseLLMInstance):
    def __init__(self):
        super().__init__(model_name="ChatGPT", model_id="openai/gpt-4o")

    def init_model(self, model_kwargs: dict = {}):
        self.model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def read_prompt(self, source_file_path: str, template_file_path: str):
        return super().read_prompt(source_file_path, template_file_path)

    def process_prompt(self, prompt: str, **llm_kwargs):
        response = self.model.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            **llm_kwargs,
        )
        return response.choices[0].message.content

    def write_response(self, filepath: str, response: str, meta: dict = {}):
        super().write_response(filepath, response, meta)


class Mixtral8x7BInstance(BaseLLMInstance):
    """
    Mixtral8x7BInstance is a class that represents an instance of the Mixtral 8x7B language model in the MLX and Transformers libraries.

    Attributes:
        model_info (dict): Information about the model.
            - display_name (str): The display name of the model.
            - name (str): The name of the model.
            - basemodel_hf_id (str): The Hugging Face ID of the base model.
            - mlx_hf_id (str): The Hugging Face ID of the quantized MLX model.
        model_id (str): The ID of the model to be used.

    Methods:
        __init__(self, use_mlx_model=True): Initializes a Mixtral8x7BInstance object.
        get_model_device(self, device_map: str = "auto"): Returns the device on which the model is initialized.
        init_model(self, model_kwargs: dict = {"torch_dtype": torch.float16}): Initializes the language model and tokenizer.
        read_prompt(self, source_file_path: str, template_file_path: str): Reads and merges source and template JSON data.
        process_prompt(self, prompt: str, **llm_kwargs): Generates a response by processing the prompt, using the language model.
        write_response(self, filepath: str, response: str, meta: dict = {}): Writes the response to a file.
    """

    def __init__(self, use_mlx_model=True):
        """
        Initialize a Mixtral8x7BInstance object.

        Parameters:
            use_mlx_model (bool, optional): Whether to use the quantized MLX model or the base model from Hugging Face. Defaults to True.

        Attributes:
            model_info (dict): Information about the model.
                - display_name (str): The display name of the model.
                - name (str): The name of the model.
                - basemodel_hf_id (str): The Hugging Face ID of the base model.
                - mlx_hf_id (str): The Hugging Face ID of the quantized MLX model.

            model_id (str): The ID of the model to be used.

        Returns:
            None
        """
        # Set model data
        self.model_info = MODEL_DEF["mixtral-8x7b"]

        # Use appropriate model from Hugging Face -> Quantized has lower precision but better performance
        model_id = (
            self.model_info["mlx_hf_id"]
            if use_mlx_model
            else self.model_info["basemodel_hf_id"]
        )

        super().__init__(model_name=self.model_info["name"], model_id=model_id)

    def get_model_device(self, device_map: str = "auto"):
        return super().get_model_device(device_map)

    def init_model(
        self,
        model_kwargs: dict = {
            "torch_dtype": torch.float16,
        },
    ):
        """
        Initializes the language model and tokenizer.

        Parameters:
            model_kwargs (dict, optional): Keyword arguments for model initialization. Defaults to {"torch_dtype": torch.float16}.

        Returns:
            None

        Notes:
            - If CUDA is available and the device is set to CUDA, the model will be initialized with 4-bit quantization configuration.
            - If CUDA is not available or the device is not CUDA, the model will be initialized with MLX configuration.
        """
        # 4-bit quantization configuration -- Not available on devices with no CUDA support -
        if torch.cuda.is_available() and self.device == torch.device("cuda"):
            nf4_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs = model_kwargs | {"quantization_config": nf4_config}

        else:
            # MLX config
            model_kwargs = model_kwargs | {
                "dim": 4096,
                "n_layers": 32,
                "head_dim": 128,
                "hidden_dim": 14336,
                "n_heads": 32,
                "n_kv_heads": 8,
                "norm_eps": 1e-05,
                "vocab_size": 32000,
                "moe": {"num_experts_per_tok": 2, "num_experts": 8},
            }

        super().init_model(model_kwargs=model_kwargs)

    def read_prompt(self, source_file_path: str, template_file_path: str):
        return super().read_prompt(source_file_path, template_file_path)

    def process_prompt(self, prompt: str, **llm_kwargs):
        """
        Process the prompt by generating a response using the language model.

        Parameters:
            prompt (str): The prompt to be processed.
            **llm_kwargs: Additional keyword arguments for generating the response.

        Returns:
            str: The generated response.

        Notes:
            - The prompt is converted into a list of messages, where each message has a role and content.
            - The tokenizer applies a chat template to the messages and converts them into input IDs.
            - If the device is set to 'mps', MLX is used for generating the response.
            - If the device is not 'mps', transformers is used for generating the response.
            - The generated response is decoded and returned as a string.
        """

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages)

        if self.device == torch.device("mps"):
            # Using MLX
            # Decode and return the generated text
            prompt = self.tokenizer.decode(input_ids)

            # Generate response using the model
            response = mlx_lm.generate(
                self.model, self.tokenizer, prompt=prompt, **llm_kwargs
            )
        else:
            # Using transformers
            # Generate response using the model
            outputs = self.model.generate(input_ids, **llm_kwargs)

            # Decode and return the generated text
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def write_response(self, filepath: str, response: str, meta: dict = {}):
        super().write_response(filepath, response, meta)


class LlamaV38BInstance(BaseLLMInstance):
    """
    LlamaV38BInstance is a class that extends the BaseLLMInstance abstract base class and provides specific functionality for the LlamaV38B language model.

    Attributes:
        model_info (dict): Information about the LlamaV38B model, including display name, name, base model Hugging Face ID, and MLX Hugging Face ID.

    Methods:
        __init__(self, use_mlx_model=True): Initializes the LlamaV38BInstance object.
        get_model_device(self, device_map: str = "auto"): Returns the device on which the model is initialized.
        init_model(self, model_kwargs: dict = {"torch_dtype": torch.bfloat16}): Initializes the language model and tokenizer.
        read_prompt(self, source_file_path: str, template_file_path: str): Reads and merges source and template JSON data.
        process_prompt(self, prompt: str, **llm_kwargs): Generates a response by processing the prompt using the LlamaV38B language model.
        write_response(self, filepath: str, response: str, meta: dict = {}): Writes the response to a file.

    """

    def __init__(self, use_mlx_model=True):
        """
        Initialize a LlamaV38BInstance object.

        Parameters:
            use_mlx_model (bool, optional): Whether to use the quantized MLX model or the base model from Hugging Face. Defaults to True.

        Attributes:
            model_info (dict): Information about the model.
                - display_name (str): The display name of the model.
                - name (str): The name of the model.
                - basemodel_hf_id (str): The Hugging Face ID of the base model.
                - mlx_hf_id (str): The Hugging Face ID of the quantized MLX model.

            model_id (str): The ID of the model to be used.

        Returns:
            None
        """
        # Set model data
        self.model_info = MODEL_DEF["meta-llama-3-8b"]

        # Use appropriate model from Hugging Face -> Quantized has lower precision but better performance
        model_id = (
            self.model_info["mlx_hf_id"]
            if use_mlx_model
            else self.model_info["basemodel_hf_id"]
        )

        super().__init__(model_name=self.model_info["name"], model_id=model_id)

    def get_model_device(self, device_map: str = "auto"):
        return super().get_model_device(device_map)

    def init_model(
        self,
        model_kwargs: dict = {
            "torch_dtype": torch.bfloat16,
        },
    ):
        super().init_model(model_kwargs=model_kwargs)

    def read_prompt(self, source_file_path: str, template_file_path: str):
        return super().read_prompt(source_file_path, template_file_path)

    def process_prompt(self, prompt: str, **llm_kwargs):
        """
        Process the prompt by generating a response using the language model.

        Parameters:
            prompt (str): The prompt to be processed.
            **llm_kwargs: Additional keyword arguments for generating the response.

        Returns:
            str: The generated response.

        Notes:
            - The prompt is converted into a list of messages, where each message has a role and content.
            - The tokenizer applies a chat template to the messages and converts them into input IDs.
            - If the device is set to 'mps', MLX is used for generating the response.
            - If the device is not 'mps', transformers is used for generating the response.
            - The generated response is decoded and returned as a string.
            - Algorithm from: https://medium.com/@xuer.chen.human/beginners-guide-to-running-llama-3-8b-on-a-macbook-air-ffb380aeef0c
        """

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages)

        if self.device == torch.device("mps"):
            # Using MLX

            # Decode and return the generated text
            prompt = self.tokenizer.decode(input_ids)
            # Generate response using the model
            response = mlx_lm.generate(
                self.model, self.tokenizer, prompt=prompt, **llm_kwargs
            )
        else:
            # Using transformers
            # Generate response using the model
            outputs = self.model.generate(input_ids, **llm_kwargs)

            # Decode and return the generated text
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def write_response(self, filepath: str, response: str, meta: dict = {}):
        super().write_response(filepath, response, meta)


if __name__ == "__main__":
    # Instantiate the model class with the desired LLM model name
    model_instance = ChatGPTInstance()

    # Process prompts given the paths to source and template JSON files
    prompt = "Provide a list of three states of the USA, only the names."
    print(f"PROMPT: \n {prompt}")
    print(f"RESPONSE: \n {model_instance.process_prompt(prompt)}")
