from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob  # For sentiment analysis
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
from abc import ABC, abstractmethod

nltk.download("punkt")
nltk.download("stopwords")


class BaseLLMBenchmark(ABC):

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.stop_words = set(stopwords.words("english"))
        self.metrics_history = []
        self.prompt_vectors = []  # Added for storing prompt vectors

    def preprocess(self, text: str):
        # Tokenize, remove stopwords, and lowercase
        tokens = word_tokenize(text)
        filtered_tokens = [
            w.lower()
            for w in tokens
            if w.lower() not in self.stop_words and w.isalnum()
        ]
        return filtered_tokens

    def vectorize_prompt(self, prompt: str):
        """Vectorize the prompt using TF-IDF and return the vector."""
        prompt_tokens = self.preprocess(prompt)
        prompt_joined = " ".join(prompt_tokens)  # Join tokens for TF-IDF vectorization
        vector = self.vectorizer.fit_transform([prompt_joined]).toarray()
        self.prompt_vectors.append(vector[0])
        return vector

    def plot_prompt_centroids(self, n_components: int = 2):
        """Plot the centroids of the vectorized prompts in 2D space using Plotly Express."""
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
        # Calculates average similarity between two sets of tokens
        similarities = []
        for token1 in tokens1:
            for token2 in tokens2:
                if token1 in model.wv and token2 in model.wv:
                    similarities.append(model.wv.similarity(token1, token2))
        return np.mean(similarities) if similarities else 0

    def evaluate_accuracy(self, expected_response: str, llm_response: str):
        # Placeholder for accuracy; consider domain-specific methods for real accuracy assessment
        return 1 / (1 + np.abs(len(expected_response) - len(llm_response)))

    def evaluate_coherence(self, response: str):
        # Use TextBlob for a simple proxy of coherence through readability (sentence length and complexity)
        blob = TextBlob(response)
        sentence_lengths = [len(sentence.words) for sentence in blob.sentences]
        average_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        coherence_score = (
            1 / (1 + np.var(sentence_lengths)) if average_sentence_length > 0 else 0
        )
        return coherence_score

    def evaluate_creativity(self, prompt_tokens: str, response_tokens: str):
        # Creativity: lexical diversity in the response not present in the prompt
        unique_response_tokens = set(response_tokens) - set(prompt_tokens)
        if not response_tokens:
            return 0
        return len(unique_response_tokens) / len(set(response_tokens))

    def evaluate_engagement(self, response):
        # TODO: Placeholder for engagement; could be enhanced with more sophisticated NLP metrics
        return len(response) / 1000  # Simple proxy based on length

    def evaluate_sentiment_alignment(self, expected_response: str, llm_response: str):
        expected_sentiment = TextBlob(expected_response).sentiment.polarity
        response_sentiment = TextBlob(llm_response).sentiment.polarity
        return 1 - abs(expected_sentiment - response_sentiment)

    def evaluate_response(
        self, prompt: str, expected_response: str, llm_response: str
    ) -> dict:
        prompt_tokens = self.preprocess(prompt)
        expected_tokens = self.preprocess(expected_response)
        response_tokens = self.preprocess(llm_response)

        # Train Word2Vec model on the combined corpus to get vectors for semantic similarity
        model = Word2Vec([prompt_tokens, expected_tokens, response_tokens], min_count=1)

        # Semantic similarity between expected and LLM response
        relevance_score = self.semantic_similarity(
            model, expected_tokens, response_tokens
        )

        results = {
            "relevance_score": relevance_score,
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

    def plot_metrics(self):
        """Plot all benchmarking metrics based on the history of evaluations using Plotly Express."""
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df = metrics_df.rename(
            columns={
                col_name: col_name.replace("_", " ").title()
                for col_name in metrics_df.columns
            }
        )
        fig = px.bar(
            metrics_df,
            y=metrics_df.columns,
            barmode="group",
            text_auto=".3f",
            title="LLM Benchmarking Metrics",
        )
        fig.update_xaxes(
            showticklabels=False,
            type="category",
        )
        fig.update_layout(
            xaxis_title="Evaluation Instance",
            yaxis_title="Scores",
            legend_title="Metrics",
        )
        fig.show()
