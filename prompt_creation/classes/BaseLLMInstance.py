# --- Imports ---
import json
import logging
import os
from abc import ABC, abstractmethod
import transformers
import mlx_lm

from dotenv import load_dotenv


# --- Classes ---
class BaseLLMInstance(ABC):
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

    def __init__(self, model_name: str, model_id: str):

        # Load environment variables
        load_dotenv()

        self.model_name = model_name
        self.model_id = model_id
        self.pipeline = None
        self.hf_token = os.getenv("HF_TOKEN")
        self.init_model()

    def init_model(
        self,
        device_map: str = "auto",
        model_kwargs: dict = {},
    ):

        logging.basicConfig(level=logging.INFO)
        # Using a pipeline for text generation with the model

        self.pipeline = transformers.pipeline(
            task="text-generation",
            model=self.model_id,
            device_map=device_map,
            model_kwargs=model_kwargs,
            token=self.hf_token,
        )

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


class AppleSiliconBaseLLMInstance(BaseLLMInstance):
    def __init__(self, model_name: str, model_id: str):
        super().__init__(model_name=model_name, model_id=model_id)

    def init_model(self):

        logging.basicConfig(level=logging.INFO)
        # Using a pipeline for text generation with the model
        self.model, self.tokenizer = mlx_lm.load(self.model_id)

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
