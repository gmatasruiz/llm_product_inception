# --- Imports ---
import json
import logging
import os
from abc import ABC, abstractmethod
import transformers
import mlx_lm
import torch

from dotenv import load_dotenv


# --- Classes ---
class BaseLLMInstance(ABC):
    """
    BaseLLMInstance is an abstract base class that provides a common interface for Language Model instances in the MLX software suite.

    Attributes:
        model_name (str): The name of the language model.
        model_id (str): The identifier of the language model.
        device (torch.device): The device on which the model is initialized.
        hf_token (str): The Hugging Face token.
        model (transformers.PreTrainedModel): The language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the language model.

    Methods:
        __init__(self, model_name: str, model_id: str): Initializes the BaseLLMInstance object.
        get_model_device(self, device_map: str = "auto"): Returns the device on which the model is initialized.
        init_model(self, model_kwargs: dict = {}): Initializes the language model and tokenizer.
        read_prompt(self, source_file_path: str, template_file_path: str): Reads and merges source and template JSON data.
        process_prompt(self, source_file_path: str, template_file_path: str, **llm_kwargs): Abstract method for processing the prompt.
        write_response(self, filepath: str, response: str): Writes the response to a file.

    """

    def __init__(self, model_name: str, model_id: str):
        """
        Initializes the BaseLLMInstance object.

        Parameters:
            model_name (str): The name of the language model.
            model_id (str): The identifier of the language model.

        Returns:
            None

        Raises:
            None
        """

        # Load environment variables
        load_dotenv()

        self.model_name = model_name
        self.model_id = model_id
        self.device = self.get_model_device()
        self.hf_token = os.getenv("HF_TOKEN")
        self.init_model()

    def get_model_device(self, device_map: str = "auto"):
        """
        Returns the device on which the model is initialized.

        Parameters:
            device_map (str, optional): The device mapping strategy. Defaults to "auto".

        Returns:
            torch.device: The device on which the model is initialized.

        Raises:
            None
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            # If device_map is 'auto', let's decide the device based on whether CUDA is available
            return torch.device(
                "cuda" if torch.cuda.is_available() and device_map == "auto" else "cpu"
            )

    def init_model(
        self,
        model_kwargs: dict = {},
    ):
        """
        Initializes the language model and tokenizer.

        Parameters:
            model_kwargs (dict, optional): Additional keyword arguments to be passed to the model initialization. Defaults to an empty dictionary.

        Returns:
            None

        Raises:
            None
        """
        logging.basicConfig(level=logging.INFO)
        if self.device == torch.device("mps"):
            """
            @Citing MLX
            The MLX software suite was initially developed with equal contribution by Awni Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. If you find MLX useful in your research and wish to cite it, please use the following BibTex entry:

            @software{mlx2023,
            author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
            title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
            url = {https://github.com/ml-explore},
            version = {0.0},
            year = {2023},
            }
            """

            self.model, self.tokenizer = mlx_lm.load(
                self.model_id, tokenizer_config=model_kwargs
            )

        else:
            # Load the tokenizer and model
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.hf_token,
                ignore_mismatched_sizes=True,
            )
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=self.hf_token,
                **model_kwargs,
                ignore_mismatched_sizes=True
            )

            self.model.to(self.device)

    def read_prompt(self, source_file_path: str, template_file_path: str):
        """
        Reads and merges source and template JSON data.

        Args:
            source_file_path (str): The path to the source JSON file, should contain the "data" field.
            template_file_path (str): The path to the template JSON file, should contain the "data" field with the "%%source" placeholder.

        Returns:
            str: The merged prompt text.
        """
        # Load the source JSON content
        with open(source_file_path, "r") as file:
            source_json = json.load(file)
        source_data = source_json["data"]

        # Load the template JSON content
        with open(template_file_path, "r") as file:
            template_json = json.load(file)
        template_data = template_json["data"]

        # Replace the placeholder with the actual source data
        merged_prompt = template_data.replace(r"%%source", source_data)

        return merged_prompt

    @abstractmethod
    def process_prompt(self, prompt: str, **llm_kwargs):
        """
        Generates a response by processing the prompt, using the language model.

        Parameters:
            prompt (str): The prompt text to be processed.
            **llm_kwargs: Additional keyword arguments specific to the language model.

        Returns:
            None

        Raises:
            NotImplementedError: This method is abstract and must be implemented in a subclass.
        """
        pass

    def write_response(self, filepath: str, response: str, meta: dict = {}):
        """
        Writes the response to a file.

        Parameters:
            filepath (str): The path to the file where the response will be written.
            response (str): The response text to be written.
            meta (dict, optional): Additional metadata to be included in the file. Defaults to an empty dictionary.

        Returns:
            None

        Raises:
            None
        """
        data_dict = {"response": response}
        meta_dict = {"__meta__": {k: v for k, v in meta.items()}}

        with open(filepath, "w") as file:
            json.dump(data_dict | meta_dict, file, indent=4)
