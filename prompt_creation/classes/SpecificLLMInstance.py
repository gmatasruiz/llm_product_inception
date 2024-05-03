# --- Imports ---
import torch
import mlx_lm


from BaseLLMInstance import BaseLLMInstance, AppleSiliconBaseLLMInstance

# --- Constants ---
MODEL_DEF = {
    "mixtral-8x7b": {
        "display_name": "Mixtral 8x7B",
        "name": "mixtral-8x-7b",
        "hf_id": "mistralai/Mixtral-8x7B-v0.1",
    },
    "applesilicon_mixtral-8x7b": {
        "display_name": "(Apple Silicon) Mixtral 8x7B",
        "name": "AppleS_mixtral-8x-7b",
        "hf_id": "mlx-community/Mixtral-8x7B-v0.1-hf-4bit-mlx",
    },
    "meta-llama-3-8b": {
        "display_name": "Meta Llama3 8B",
        "name": "Meta-Llama-3-8B",
        "hf_id": "meta-llama/Meta-Llama-3-8B",
    },
    "applesilicon_meta-llama-3-8b": {
        "display_name": "(Apple Silicon) Meta Llama3 8B",
        "name": "AppleS_Meta-Llama-3-8B",
        "hf_id": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
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
    def __init__(self):
        self.model_info = MODEL_DEF["mixtral-8x7b"]
        super().__init__(
            model_name=self.model_info["name"], model_id=self.model_info["hf_id"]
        )

    def init_model(
        self,
        model_kwargs: dict = {
            "torch_dtype": torch.float16,
        },
    ):

        # 4-bit quantization configuration -- Not available on devices with no CUDA support -
        # nf4_config = transformers.BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )
        # model_kwargs = model_kwargs | {"quantization_config": nf4_config}

        super().init_model(model_kwargs=model_kwargs)

    def read_prompt(self, source_file_path: str, template_file_path: str):
        return super().read_prompt(source_file_path, template_file_path)

    def process_prompt(self, prompt: str, **llm_kwargs):
        response = self.pipeline(prompt, **llm_kwargs)[0]["generated_text"]
        return response

    def write_response(self, filepath: str, response: str):
        super().write_response(filepath, response)


class LlamaV38BInstance(BaseLLMInstance):
    def __init__(self):
        self.model_info = MODEL_DEF["meta-llama-3-8b"]
        super().__init__(
            model_name=self.model_info["name"], model_id=self.model_info["hf_id"]
        )

    def init_model(
        self,
        model_kwargs: dict = {"torch_dtype": torch.bfloat16},
    ):
        super().init_model(model_kwargs=model_kwargs)

    def read_prompt(self, source_file_path: str, template_file_path: str):
        return super().read_prompt(source_file_path, template_file_path)

    def process_prompt(self, prompt: str, **llm_kwargs):
        response = self.pipeline(prompt, **llm_kwargs)[0]["generated_text"]
        return response

    def write_response(self, filepath: str, response: str):
        super().write_response(filepath, response)


class AppleSiliconMixtral8x7BInstance(AppleSiliconBaseLLMInstance):
    def __init__(self):
        self.model_info = MODEL_DEF["applesilicon_mixtral-8x7b"]
        super().__init__(
            model_name=self.model_info["name"], model_id=self.model_info["hf_id"]
        )

    def init_model(self):
        super().init_model()

    def read_prompt(self, source_file_path: str, template_file_path: str):
        return super().read_prompt(source_file_path, template_file_path)

    def process_prompt(self, prompt: str, **llm_kwargs):
        # From: https://medium.com/@xuer.chen.human/beginners-guide-to-running-llama-3-8b-on-a-macbook-air-ffb380aeef0c
        messages = [
            {"role": "user", "content": prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(messages)
        prompt_ = self.tokenizer.decode([input_ids[0]])
        response = mlx_lm.generate(
            self.model, self.tokenizer, prompt=prompt, **llm_kwargs
        )
        return response

    def write_response(self, filepath: str, response: str):
        super().write_response(filepath, response)


class AppleSiliconLlamaV38BInstance(AppleSiliconBaseLLMInstance):
    def __init__(self):
        self.model_info = MODEL_DEF["applesilicon_meta-llama-3-8b"]
        super().__init__(
            model_name=self.model_info["name"], model_id=self.model_info["hf_id"]
        )

    def init_model(self):
        super().init_model()

    def read_prompt(self, source_file_path: str, template_file_path: str):
        return super().read_prompt(source_file_path, template_file_path)

    def process_prompt(self, prompt: str, **llm_kwargs):
        # From: https://medium.com/@xuer.chen.human/beginners-guide-to-running-llama-3-8b-on-a-macbook-air-ffb380aeef0c

        messages = [
            {"role": "user", "content": prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(messages)
        prompt_ = self.tokenizer.decode([input_ids[0]])
        response = mlx_lm.generate(
            self.model, self.tokenizer, prompt=prompt, **llm_kwargs
        )
        return response

    def write_response(self, filepath: str, response: str):
        super().write_response(filepath, response)


if __name__ == "__main__":
    # Instantiate the model class with the desired LLM model name
    model_instance = AppleSiliconMixtral8x7BInstance()

    # Process prompts given the paths to source and template JSON files
    prompt = "Provide a list of three states of the USA, only the names."
    print(f"PROMPT: \n {prompt}")
    print(
        f"RESPONSE: \n {model_instance.process_prompt(prompt, temp=0.01, max_tokens=24, verbose=True)}"
    )
