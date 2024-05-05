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

    def __init__(self, model_name: str, model_id: str):

        # Load environment variables
        load_dotenv()

        self.model_name = model_name
        self.model_id = model_id
        self.device = self.get_model_device()
        self.hf_token = os.getenv("HF_TOKEN")
        self.init_model()

    def get_model_device(self, device_map: str = "auto"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            # If device_map is 'auto', let's decide the device based on whether CUDA is available
            return torch.device(
                "cuda"
                if torch.cuda.is_available() and device_map == "auto"
                else "cpu"
            )

    def init_model(
        self,
        model_kwargs: dict = {},
    ):
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
            source_file_path (str): The path to the source JSON file.
            template_file_path (str): The path to the template JSON file.

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
    def process_prompt(
        self, source_file_path: str, template_file_path: str, **llm_kwargs
    ):
        pass

    def write_response(self, filepath: str, response: str):
        with open(filepath, "w") as file:
            json.dump({"response": response}, file, indent=4)
