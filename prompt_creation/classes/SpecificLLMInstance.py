# --- Imports ---
import torch
import transformers
import mlx_lm


from classes.BaseLLMInstance import BaseLLMInstance


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
        NotImplemented

    def init_model(
        self,
        device_map: str = "auto",
        model_kwargs: dict = {},
    ):
        NotImplemented

    def read_prompt(self, source_file_path: str, template_file_path: str):
        NotImplemented

    def process_prompt(self, prompt: str):
        NotImplemented

    def write_response(self, filepath: str, response: str):
        NotImplemented


class Mixtral8x7BInstance(BaseLLMInstance):
    def __init__(self, use_mlx_model=True):
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

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages)

        if self.device == torch.device("mps"):
            # Using MLX
            prompt = self.tokenizer.decode(input_ids)
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

    def write_response(self, filepath: str, response: str):
        super().write_response(filepath, response)


class LlamaV38BInstance(BaseLLMInstance):
    def __init__(self, use_mlx_model=True):
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
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages)

        if self.device == torch.device("mps"):
            # Using MLX
            # From: https://medium.com/@xuer.chen.human/beginners-guide-to-running-llama-3-8b-on-a-macbook-air-ffb380aeef0c
            prompt = self.tokenizer.decode(input_ids)
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

    def write_response(self, filepath: str, response: str):
        super().write_response(filepath, response)


if __name__ == "__main__":
    # Instantiate the model class with the desired LLM model name
    model_instance = LlamaV38BInstance(use_mlx_model=True)

    # Process prompts given the paths to source and template JSON files
    prompt = "Provide a list of three states of the USA, only the names."
    print(f"PROMPT: \n {prompt}")
    print(
        f"RESPONSE: \n {model_instance.process_prompt(prompt, temp=0.01, max_tokens=100, verbose=True)}"
    )
