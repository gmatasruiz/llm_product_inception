# --- Imports ---
import os
import json

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob  # For sentiment analysis
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = None

from abc import ABC
from transformers import BertTokenizer, BertModel  # For BERT embeddings


nltk.download("punkt")
nltk.download("stopwords")
# Ensure BERT model and tokenizer are set up
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")


# --- Classes ---
class BaseLLMBenchmark(ABC):
    """
    BaseLLMBenchmark class for evaluating the performance of a Language Model (LLM).

    This class provides methods for preprocessing text, vectorizing prompts, plotting prompt centroids, calculating semantic similarity, vectorizing text using BERT, evaluating accuracy, coherence, creativity, engagement, and sentiment alignment of LLM responses. It also includes a method for evaluating the overall response based on multiple metrics and plotting the evaluation metrics history.

    Attributes:
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer used for vectorizing prompts.
        stop_words (set): A set of stopwords for text preprocessing.
        metrics_history (list): A list to store the evaluation results of LLM responses.
        prompt_vectors (list): A list to store the vectorized representations of prompts.

    Methods:
        preprocess(text: str) -> list:
            Tokenizes the input text, removes stopwords, and converts the tokens to lowercase.

        vectorize_prompt(prompt: str) -> numpy.ndarray:
            Vectorizes the prompt using TF-IDF and returns the vector.

        plot_prompt_centroids(n_components: int = 2) -> None:
            Plots the centroids of the vectorized prompts in 2D or 3D space using Plotly Express.

        semantic_similarity(model: Word2Vec, tokens1: list[str], tokens2: list[str]) -> float:
            Calculates the average similarity between two sets of tokens using a Word2Vec model, enhanced with
            sentiment alignment for a deeper semantic comparison.

        bert_vectorize(text: str) -> numpy.ndarray:
            Vectorizes the input text using BERT and returns the vector representation.

        evaluate_accuracy(expected_response: str, llm_response: str) -> float:
            Calculates the accuracy score between the expected response and the LLM response using cosine similarity.

        evaluate_coherence(response: str) -> float:
            Calculates the coherence score of a response based on sentence length and complexity.

        evaluate_creativity(prompt_tokens: str, response_tokens: str) -> float:
            Calculates the creativity score of a response based on the lexical diversity not present in the prompt.

        evaluate_engagement(response: str) -> float:
            Calculates the engagement score of a response based on sentiment polarity, subjectivity, and normalized length.

        evaluate_sentiment_alignment(expected_response: str, llm_response: str) -> float:
            Evaluates the sentiment alignment between the expected response and the generated response by the LLM.

        evaluate_response(prompt: str, expected_response: str, llm_response: str) -> dict:
            Evaluates the response generated by the LLM based on multiple metrics and returns the results as a dictionary.

        plot_metrics() -> None:
            Plots all benchmarking metrics based on the history of evaluations using Plotly Express.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.stop_words = set(stopwords.words("english"))
        self.metrics_history = []
        self.prompt_vectors = []  # Added for storing prompt vectors

    def preprocess(self, text: str):
        """
        Tokenizes the input text, removes stopwords, and converts the tokens to lowercase.

        Parameters:
            text (str): The input text to be preprocessed.

        Returns:
            list: A list of filtered tokens after preprocessing.

        """
        # Tokenize, remove stopwords, and lowercase
        tokens = word_tokenize(text)
        filtered_tokens = [
            w.lower()
            for w in tokens
            if w.lower() not in self.stop_words and w.isalnum()
        ]
        return filtered_tokens

    def vectorize_prompt(self, prompt: str):
        """
        Vectorize the prompt using TF-IDF and return the vector.

        Parameters:
            prompt (str): The prompt to be vectorized.

        Returns:
            numpy.ndarray: The vectorized representation of the prompt.

        """
        prompt_tokens = self.preprocess(prompt)
        prompt_joined = " ".join(prompt_tokens)  # Join tokens for TF-IDF vectorization
        vector = self.vectorizer.fit_transform([prompt_joined]).toarray()
        self.prompt_vectors.append(vector[0])
        return vector

    def plot_prompt_centroids(self, n_components: int = 2):
        """
        Plot the centroids of the vectorized prompts in 2D or 3D space using Plotly Express.

        Parameters:
            n_components (int, optional): The number of dimensions to reduce the vectorized prompts to. Defaults to 2.

        Returns:
            None

        Raises:
            None

        Notes:
            - This method requires that the prompts have already been vectorized using the 'vectorize_prompt' method.
            - If the 'prompt_vectors' attribute is empty, a message will be printed indicating that no prompts have been vectorized yet.
            - If the 'n_components' parameter is not set to 2 or 3, a message will be printed indicating that the dimensions are not correct.

        Example:
            benchmark = BaseLLMBenchmark()
            benchmark.vectorize_prompt("This is a prompt.")
            benchmark.vectorize_prompt("This is another prompt.")
            benchmark.plot_prompt_centroids(n_components=2)
        """

        if not self.prompt_vectors:
            print("No prompts have been vectorized yet.")
            return

        if n_components not in [2, 3]:
            print("Dimensions are not correct, please set to `2` or `3`")
            return

        # Convert the list of vectors to a matrix for PCA
        vectors_matrix = np.array(self.prompt_vectors)

        # Standardize the features
        vectors_matrix = StandardScaler().fit_transform(vectors_matrix)

        # PCA to reduce to 2 dimensions
        pca = PCA(n_components)
        principalComponents = pca.fit_transform(vectors_matrix)
        pca_columns = [f"PC {i+1}" for i in range(n_components)]

        # Convert to a DataFrame for ease of plotting
        df = pd.DataFrame(data=principalComponents, columns=pca_columns)

        # Plotting using Plotly Express
        if n_components == 2:
            fig = px.scatter(
                df,
                x=df.columns[0],
                y=df.columns[1],
                title="Centroids of Vectorized Prompts",
            )
        elif n_components == 3:
            fig = px.scatter_3d(
                df,
                x=df.columns[0],
                y=df.columns[1],
                z=df.columns[2],
                title="Centroids of Vectorized Prompts",
            )

        fig.update_traces(marker=dict(size=12, opacity=0.5))
        fig.show()

    def semantic_similarity(
        self, model: Word2Vec, tokens1: list[str], tokens2: list[str]
    ):
        """
        Calculates the average similarity between two sets of tokens using a Word2Vec model,
        enhanced with sentiment alignment for a deeper semantic comparison.

        Parameters:
            model (Word2Vec): The Word2Vec model used for calculating the similarity.
            tokens1 (list[str]): The first set of tokens.
            tokens2 (list[str]): The second set of tokens.

        Returns:
            float: The weighted average similarity between the two sets of tokens,
            incorporating sentiment alignment. If no similarity is found, returns 0.
        """
        # Parameter weights
        W_POLARITY, W_SIMILARITY = (0.3, 0.7)

        # Calculate average similarity based on Word2Vec model
        similarities = []
        for token1 in tokens1:
            for token2 in tokens2:
                if token1 in model.wv and token2 in model.wv:
                    # Compute similarity score
                    sim_score = model.wv.similarity(token1, token2)
                    similarities.append(sim_score)

        if not similarities:
            return 0

        # Convert tokens into full text to analyze sentiment
        text1 = " ".join(tokens1)
        text2 = " ".join(tokens2)

        # Use TextBlob to calculate sentiment polarity of each text
        sentiment1 = TextBlob(text1).sentiment.polarity
        sentiment2 = TextBlob(text2).sentiment.polarity

        # Calculate sentiment alignment as the absolute difference in polarity
        sentiment_alignment = 1 - abs(sentiment1 - sentiment2)

        # Weighting: 70% weight to semantic similarity, 30% to sentiment alignment
        weighted_similarity = (
            sentiment_alignment * W_POLARITY + np.mean(similarities) * W_SIMILARITY
        )

        return weighted_similarity

    def bert_vectorize(self, text: str):
        """
        Vectorize the input text using BERT and return the vector representation.

        Parameters:
            text (str): The input text to be vectorized.

        Returns:
            numpy.ndarray: The vector representation of the input text.

        Notes:
            - This method uses the BERT tokenizer and model to encode the input text.
            - The input text is tokenized, padded, and truncated using the BERT tokenizer.
            - The BERT model is used to generate the last hidden state of the encoded input.
            - The mean of the last hidden state is taken along the sequence dimension to obtain a fixed-length vector representation.
            - The vector representation is converted to a numpy array and returned.

        Example:
            benchmark = BaseLLMBenchmark()
            vector = benchmark.bert_vectorize("This is an example text.")
            print(vector)
        """
        encoded_input = bert_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            model_output = bert_model(**encoded_input)
        return model_output.last_hidden_state.mean(dim=1).squeeze().numpy()

    def evaluate_accuracy(self, expected_response: str, llm_response: str):
        """
        Calculate the accuracy score between the expected response and the LLM response using cosine similarity.

        Parameters:
            expected_response (str): The expected response.
            llm_response (str): The LLM response.

        Returns:
            float: The accuracy score, which is a value between -1 and 1. A higher score indicates a higher similarity between the expected response and the LLM response.

        Notes:
            - This method uses the BERT model to vectorize the expected response and the LLM response.
            - The BERT model generates embeddings for the input text, which are then used to calculate the cosine similarity.
            - The cosine similarity measures the similarity between two vectors by calculating the cosine of the angle between them.
            - The accuracy score is calculated as the cosine similarity between the expected response embedding and the LLM response embedding.

        Example:
            benchmark = BaseLLMBenchmark()
            expected_response = "This is the expected response."
            llm_response = "This is the LLM response."
            accuracy_score = benchmark.evaluate_accuracy(expected_response, llm_response)
            print(accuracy_score)
        """
        expected_embedding = self.bert_vectorize(expected_response)
        response_embedding = self.bert_vectorize(llm_response)
        accuracy_score = cosine_similarity([expected_embedding], [response_embedding])[
            0, 0
        ]
        return accuracy_score

    def evaluate_coherence(self, response: str):
        """
        Calculate the coherence score of a response based on sentence length and complexity.

        This method uses TextBlob to analyze the response and calculate the coherence score.
        The coherence score is a measure of how well-structured and coherent the response is,
        based on the average sentence length and complexity.

        Parameters:
            response (str): The response to evaluate.

        Returns:
            float: The coherence score, which is a value between 0 and 1. A higher score indicates a more coherent response.

        Notes:
            - The coherence score is calculated using the formula: 1 / (1 + variance of sentence lengths).
            - If the average sentence length is 0, the coherence score will be 0.

        Example:
            benchmark = BaseLLMBenchmark()
            coherence_score = benchmark.evaluate_coherence("This is a coherent response.")
            print(coherence_score)
        """
        # Use TextBlob for a simple proxy of coherence through readability (sentence length and complexity)
        blob = TextBlob(response)
        sentence_lengths = [len(sentence.words) for sentence in blob.sentences]
        average_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        coherence_score = (
            1 / (1 + np.var(sentence_lengths)) if average_sentence_length > 0 else 0
        )
        return coherence_score

    def evaluate_creativity(self, prompt_tokens: str, response_tokens: str):
        """
        Calculate the creativity score of a response based on the lexical diversity not present in the prompt.

        This method calculates the creativity score by measuring the lexical diversity of the response that
        is not present in the prompt. It calculates the ratio of unique tokens in the response that are not
        present in the prompt to the total number of unique tokens in the response.

        Parameters:
            prompt_tokens (str): The tokens of the prompt.
            response_tokens (str): The tokens of the response.

        Returns:
            float: The creativity score, which is a value between 0 and 1. A higher score indicates a more creative response.

        Notes:
            - The creativity score is calculated using the formula: len(unique_response_tokens) / len(set(response_tokens)).
            - If the response_tokens are empty, the creativity score will be 0.

        Example:
            benchmark = BaseLLMBenchmark()
            prompt_tokens = benchmark.preprocess("This is the prompt.")
            response_tokens = benchmark.preprocess("This is a creative response.")
            creativity_score = benchmark.evaluate_creativity(prompt_tokens, response_tokens)
            print(creativity_score)
        """
        # Creativity: lexical diversity in the response not present in the prompt
        unique_response_tokens = set(response_tokens) - set(prompt_tokens)
        if not response_tokens:
            return 0
        return len(unique_response_tokens) / len(set(response_tokens))

    def evaluate_engagement(self, response: str):
        """
        Calculate the engagement score of a response based on sentiment polarity, subjectivity, and normalized length.

        This implementation uses sentiment analysis to gauge the emotional depth and subjectivity of the response,
        combining these with the response's length to assess overall engagement.
        The weights (0.3, 0.3, 0.4) can be adjusted based on further empirical analysis to balance the contributions of each factor.

        Parameters:
            response (str): The response to evaluate.

        Returns:
            float: The engagement score, which is a weighted sum of sentiment polarity, subjectivity, and normalized length.

        """
        # Parameter weights
        W_POLARITY, W_SUBJECTIVITY, W_NORMLENGTH = (0.3, 0.3, 0.4)

        # Use TextBlob to get sentiment polarity and subjectivity to indicate emotional engagement
        sentiment = TextBlob(response).sentiment
        polarity = abs(sentiment.polarity)  # Emotional intensity
        subjectivity = sentiment.subjectivity  # Personal engagement

        # Calculate engagement score as weighted sum of polarity, subjectivity, and normalized length
        normalized_length = (
            len(response) / 1000
        )  # Normalize by a suitable factor, like average length
        engagement_score = (
            W_POLARITY * polarity
            + W_SUBJECTIVITY * subjectivity
            + W_NORMLENGTH * normalized_length
        )
        return engagement_score

    def evaluate_sentiment_alignment(self, expected_response: str, llm_response: str):
        """
        Evaluate the sentiment alignment between the expected response and the generated response by the Language Model (LLM).

        Parameters:
            expected_response (str): The expected response.
            llm_response (str): The generated response by the LLM.

        Returns:
            float: The sentiment alignment score between the expected response and the LLM response.

        Notes:
            - This method calculates the sentiment polarity and subjectivity of both the expected response and the LLM response using the TextBlob library.
            - The sentiment alignment score is calculated as the Euclidean distance between the polarity and subjectivity scores of the expected response and the LLM response.
            - The sentiment alignment score ranges from 0 to 1, where 0 indicates perfect alignment and 1 indicates no alignment.

        Example:
            benchmark = BaseLLMBenchmark()
            expected_response = "I love this product!"
            llm_response = "I hate this product!"
            sentiment_alignment_score = benchmark.evaluate_sentiment_alignment(expected_response, llm_response)
            print(sentiment_alignment_score)
        """

        expected_sentiment = TextBlob(expected_response).sentiment
        llm_sentiment = TextBlob(llm_response).sentiment

        # Calculate the Euclidean distance between the sentiment scores
        sentiment_diff = np.linalg.norm(
            [
                expected_sentiment.polarity - llm_sentiment.polarity,
            ]
        )

        # Normalize the sentiment difference to a score between 0 and 1
        max_diff = np.sqrt(2)
        normalized_diff = sentiment_diff / max_diff
        # Maximum possible distance when both polarity and subjectivity are 1
        sentiment_alignment_score = abs(1 - 2 * (normalized_diff))

        return sentiment_alignment_score

    def evaluate_response(
        self, prompt: str, expected_response: str, llm_response: str
    ) -> dict:
        """
        Evaluate the response generated by the Language Model (LLM) based on multiple metrics and return the results as a dictionary.

        Parameters:
            prompt (str): The prompt given to the LLM.
            expected_response (str): The expected response.
            llm_response (str): The generated response by the LLM.

        Returns:
            dict: A dictionary containing the evaluation results with the following keys:
                - "relevance_score": The semantic similarity score between the expected response and the LLM response.
                - "accuracy_score": The cosine similarity score between the vectorized expected response and the vectorized LLM response.
                - "coherence_score": The coherence score of the LLM response based on sentence length and complexity.
                - "creativity_score": The creativity score of the LLM response based on the lexical diversity not present in the prompt.
                - "engagement_score": The engagement score of the LLM response based on sentiment polarity, subjectivity, and normalized length.
                - "sentiment_alignment": The sentiment alignment score between the expected response and the LLM response.

        Notes:
            - The method preprocesses the prompt, expected response, and LLM response to obtain tokens for further analysis.
            - It trains a Word2Vec model on the combined corpus of prompt, expected response, and LLM response to calculate semantic similarity.
            - The evaluation results are stored in the metrics_history attribute for plotting.

        Example:
            benchmark = BaseLLMBenchmark()
            prompt = "Please write a summary of the given article."
            expected_response = "The article discusses the impact of climate change on biodiversity."
            llm_response = "The article talks about how climate change affects the diversity of living organisms."
            results = benchmark.evaluate_response(prompt, expected_response, llm_response)
            print(results)
        """
        prompt_tokens = self.preprocess(prompt)
        expected_tokens = self.preprocess(expected_response)
        response_tokens = self.preprocess(llm_response)

        # Train Word2Vec model on the combined corpus to get vectors for semantic similarity
        model = Word2Vec([prompt_tokens, expected_tokens, response_tokens], min_count=1)

        results = {
            "semantic_similarity": self.semantic_similarity(
                model, expected_tokens, response_tokens
            ),
            "accuracy_score": self.evaluate_accuracy(expected_response, llm_response),
            "coherence_score": self.evaluate_coherence(llm_response),
            "creativity_score": self.evaluate_creativity(
                prompt_tokens, response_tokens
            ),
            "engagement_score": self.evaluate_engagement(llm_response),
            "sentiment_alignment": self.evaluate_sentiment_alignment(
                expected_response, llm_response
            ),
        }

        # Store results for plotting
        self.metrics_history.append(results)
        return results

    def write_metrics_df(self, write_to_json_file: bool = False, path: str = None):
        """
        Write the evaluation metrics history to a DataFrame and optionally to a JSON file.

        Parameters:
            write_to_json_file (bool, optional): A flag indicating whether to write the metrics to a JSON file. Defaults to False.
            path (str, optional): The path to the JSON file where the metrics will be written. Required if write_to_json_file is True.

        Returns:
            pandas.DataFrame: A DataFrame containing the evaluation metrics history.

        Raises:
            FileNotFoundError: If the specified path does not exist when trying to write to a JSON file.

        Notes:
            - The method creates a DataFrame from the stored metrics history.
            - If write_to_json_file is True and a valid path is provided, the metrics will be written to a JSON file.
            - If the path does not exist when trying to write to a JSON file, a FileNotFoundError will be raised.

        Example:
            benchmark = BaseLLMBenchmark()
            benchmark.evaluate_response("Prompt 1", "Expected Response 1", "LLM Response 1")
            benchmark.evaluate_response("Prompt 2", "Expected Response 2", "LLM Response 2")
            metrics_df = benchmark.write_metrics_df(write_to_json_file=True, path="metrics.json")
        """
        # Create dataframe from metrics history
        metrics_df = pd.DataFrame(self.metrics_history)

        if write_to_json_file and os.path.exists(os.path.dirname(path)):
            # Write metrics to json file
            metrics_df.to_json(path)

        return metrics_df

    def write_metrics_fig(self):
        """
        Generates a bar plot visualizing the benchmarking metrics.

        Returns:
            plotly.graph_objects.Figure: A bar plot displaying the benchmarking metrics with evaluation instances on the x-axis and scores on the y-axis.
                The bars are grouped by different metrics, showing the comparison of scores across evaluations.

        Notes:
            - This method internally calls 'write_metrics_df' to obtain the dataframe containing the benchmarking metrics.
            - The column names of the metrics dataframe are modified for better readability in the plot.
            - The bar plot is created using Plotly Express with grouped bars for each metric.
            - The x-axis represents the evaluation instances, while the y-axis represents the scores of the metrics.
            - The legend displays the names of the metrics being compared in the plot.

        Example:
            benchmark = BaseLLMBenchmark()
            fig = benchmark.write_metrics_fig()
            fig.show()
        """
        metrics_df = self.write_metrics_df()

        metrics_df = metrics_df.rename(
            columns={
                col_name: col_name.replace("_", " ").title()
                for col_name in metrics_df.columns
            }
        )
        fig = px.bar(
            metrics_df,
            y=metrics_df.columns,
            orientation="v",
            barmode="group",
            text_auto=".3f",
            title="LLM Benchmarking Metrics",
        )
        fig.update_xaxes(
            type="category",
        )
        fig.update_layout(
            xaxis_title="Template Number",
            yaxis_title="Scores",
            legend_title="Metrics",
        )
        return fig

    def save_metrics_to_file(
        self,
        output_fname: str,
        metrics_path: str,
        figures_path: str,
    ):

        # self.write_metrics_df(
        #     write_to_json_file=True,
        #     path=os.path.join(metrics_path, f"table_{output_fname}.json"),
        # )

        fig = self.write_metrics_fig()

        # fig.write_image(os.path.join(figures_path, f"{output_fname}.png"))
        # fig.write_html(os.path.join(figures_path, f"{output_fname}.html"))
        fig.write_json(os.path.join(figures_path, f"{output_fname}.json"))

    def plot_metrics(self):
        """
        Plot all benchmarking metrics based on the history of evaluations using Plotly Express.

        This method takes the evaluation metrics history stored in the 'metrics_history' attribute and plots them.
        The metrics are displayed as a bar chart, with each metric represented as a separate bar.
        The x-axis represents the evaluation instances, while the y-axis represents the scores of the metrics.

        Parameters:
            None

        Returns:
            None

        Example:
            benchmark = BaseLLMBenchmark()
            benchmark.plot_metrics()
        """
        fig = self.write_metrics_fig()

        fig.show()
