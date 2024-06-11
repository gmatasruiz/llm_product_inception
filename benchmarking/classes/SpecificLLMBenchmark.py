# --- Imports ---
from benchmarking.classes.BaseLLMBenchmark import BaseLLMBenchmark


# --- Classes ---
class ChatGPTBenchmark(BaseLLMBenchmark):
    """
    ChatGPTBenchmark class for evaluating the performance of ChatGPT Language Model (LLM).

    This class extends the BaseLLMBenchmark class and inherits its methods for preprocessing text, vectorizing prompts,
    plotting prompt centroids, calculating semantic similarity, vectorizing text using BERT, evaluating accuracy,
    coherence, creativity, engagement, and sentiment alignment of LLM responses. It also includes a method for evaluating
    the overall response based on multiple metrics and plotting the evaluation metrics history.
    """

    def __init__(self):
        super().__init__()
        self.llm_model = "ChatGPT"

    def preprocess(self, text: str):
        return super().preprocess(text)

    def vectorize_prompt(self, prompt: str):
        return super().vectorize_prompt(prompt)

    def plot_prompt_centroids(self, n_components: int = 2):
        super().plot_prompt_centroids(n_components)

    def semantic_similarity(self, model, tokens1: list[str], tokens2: list[str]):
        return super().semantic_similarity(model, tokens1, tokens2)

    def bert_vectorize(self, text: str):
        return super().bert_vectorize(text)

    def evaluate_accuracy(self, expected_response: str, llm_response: str):
        return super().evaluate_accuracy(expected_response, llm_response)

    def evaluate_coherence(self, response: str):
        return super().evaluate_coherence(response)

    def evaluate_creativity(self, prompt_tokens: str, response_tokens: str):
        return super().evaluate_creativity(prompt_tokens, response_tokens)

    def evaluate_engagement(self, expected_response: str, llm_response: str):
        return super().evaluate_engagement(expected_response, llm_response)

    def evaluate_sentiment_alignment(self, expected_response: str, llm_response: str):
        return super().evaluate_sentiment_alignment(expected_response, llm_response)

    def evaluate_response(
        self, prompt: str, expected_response: str, llm_response: str
    ) -> dict:
        return super().evaluate_response(prompt, expected_response, llm_response)

    def write_metrics_df(self, write_to_json_file: bool = False, path: str = None):
        return super().write_metrics_df(write_to_json_file, path)

    def write_metrics_fig(self):
        return super().write_metrics_fig()

    def plot_metrics(self):
        super().plot_metrics()


class Mixtral8x7BBenchmark(BaseLLMBenchmark):
    """
    Mixtral8x7BBenchmark class for evaluating the performance of the Mixtral 8x7B Language Model (LLM).

    This class extends the BaseLLMBenchmark class and inherits its methods for preprocessing text, vectorizing prompts,
    plotting prompt centroids, calculating semantic similarity, vectorizing text using BERT, evaluating accuracy,
    coherence, creativity, engagement, and sentiment alignment of LLM responses. It also includes a method for evaluating
    the overall response based on multiple metrics and plotting the evaluation metrics history.
    """

    def __init__(self):
        super().__init__()
        self.llm_model = "Mixtral-8x7B"

    def preprocess(self, text: str):
        return super().preprocess(text)

    def vectorize_prompt(self, prompt: str):
        return super().vectorize_prompt(prompt)

    def plot_prompt_centroids(self, n_components: int = 2):
        super().plot_prompt_centroids(n_components)

    def semantic_similarity(self, model, tokens1: list[str], tokens2: list[str]):
        return super().semantic_similarity(model, tokens1, tokens2)

    def bert_vectorize(self, text: str):
        return super().bert_vectorize(text)

    def evaluate_accuracy(self, expected_response: str, llm_response: str):
        return super().evaluate_accuracy(expected_response, llm_response)

    def evaluate_coherence(self, response: str):
        return super().evaluate_coherence(response)

    def evaluate_creativity(self, prompt_tokens: str, response_tokens: str):
        return super().evaluate_creativity(prompt_tokens, response_tokens)

    def evaluate_engagement(self, expected_response: str, llm_response: str):
        return super().evaluate_engagement(expected_response, llm_response)

    def evaluate_sentiment_alignment(self, expected_response: str, llm_response: str):
        return super().evaluate_sentiment_alignment(expected_response, llm_response)

    def evaluate_response(
        self, prompt: str, expected_response: str, llm_response: str
    ) -> dict:
        return super().evaluate_response(prompt, expected_response, llm_response)

    def write_metrics_df(self, write_to_json_file: bool = False, path: str = None):
        return super().write_metrics_df(write_to_json_file, path)

    def write_metrics_fig(self):
        return super().write_metrics_fig()

    def plot_metrics(self):
        super().plot_metrics()


class LlamaV38BBenchmark(BaseLLMBenchmark):
    """
    LlamaV38BBenchmark class for evaluating the performance of the LlamaV3 8B Language Model (LLM).

    This class extends the BaseLLMBenchmark class and inherits its methods for preprocessing text, vectorizing prompts,
    plotting prompt centroids, calculating semantic similarity, vectorizing text using BERT, evaluating accuracy,
    coherence, creativity, engagement, and sentiment alignment of LLM responses. It also includes a method for evaluating
    the overall response based on multiple metrics and plotting the evaluation metrics history.
    """

    def __init__(self):
        super().__init__()
        self.llm_model = "Meta-Llama3-8B"

    def preprocess(self, text: str):
        return super().preprocess(text)

    def vectorize_prompt(self, prompt: str):
        return super().vectorize_prompt(prompt)

    def plot_prompt_centroids(self, n_components: int = 2):
        super().plot_prompt_centroids(n_components)

    def semantic_similarity(self, model, tokens1: list[str], tokens2: list[str]):
        return super().semantic_similarity(model, tokens1, tokens2)

    def bert_vectorize(self, text: str):
        return super().bert_vectorize(text)

    def evaluate_accuracy(self, expected_response: str, llm_response: str):
        return super().evaluate_accuracy(expected_response, llm_response)

    def evaluate_coherence(self, response: str):
        return super().evaluate_coherence(response)

    def evaluate_creativity(self, prompt_tokens: str, response_tokens: str):
        return super().evaluate_creativity(prompt_tokens, response_tokens)

    def evaluate_engagement(self, expected_response: str, llm_response: str):
        return super().evaluate_engagement(expected_response, llm_response)

    def evaluate_sentiment_alignment(self, expected_response: str, llm_response: str):
        return super().evaluate_sentiment_alignment(expected_response, llm_response)

    def evaluate_response(
        self, prompt: str, expected_response: str, llm_response: str
    ) -> dict:
        return super().evaluate_response(prompt, expected_response, llm_response)

    def write_metrics_df(self, write_to_json_file: bool = False, path: str = None):
        return super().write_metrics_df(write_to_json_file, path)

    def write_metrics_fig(self):
        return super().write_metrics_fig()

    def plot_metrics(self):
        super().plot_metrics()
