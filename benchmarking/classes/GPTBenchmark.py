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
import matplotlib.pyplot as plt

nltk.download("punkt")
nltk.download("stopwords")


class ChatGPTBenchmark:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.stop_words = set(stopwords.words("english"))
        self.metrics_history = []

    def preprocess(self, text):
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

    def plot_prompt_centroids(self):
        """Plot the centroids of the vectorized prompts in 2D space."""
        if not self.prompt_vectors:
            print("No prompts have been vectorized yet.")
            return

        # Convert the list of vectors to a matrix for PCA
        vectors_matrix = np.array(self.prompt_vectors)

        # Standardize the features
        vectors_matrix = StandardScaler().fit_transform(vectors_matrix)

        # PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(vectors_matrix)

        # Convert to a DataFrame for ease of plotting
        df = pd.DataFrame(data=principalComponents, columns=["PC 1", "PC 2"])

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(df["PC 1"], df["PC 2"], s=50, alpha=0.5)
        plt.title("Centroids of Vectorized Prompts")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True)
        plt.show()

    def semantic_similarity(self, model, tokens1, tokens2):
        # Calculates average similarity between two sets of tokens
        similarities = []
        for token1 in tokens1:
            for token2 in tokens2:
                if token1 in model.wv and token2 in model.wv:
                    similarities.append(model.wv.similarity(token1, token2))
        return np.mean(similarities) if similarities else 0

    def evaluate_accuracy(self, expected_response, chatgpt_response):
        # Placeholder for accuracy; consider domain-specific methods for real accuracy assessment
        return 1 / (1 + np.abs(len(expected_response) - len(chatgpt_response)))

    def evaluate_coherence(self, response):
        # Use TextBlob for a simple proxy of coherence through readability (sentence length and complexity)
        blob = TextBlob(response)
        sentence_lengths = [len(sentence.words) for sentence in blob.sentences]
        average_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        coherence_score = (
            1 / (1 + np.var(sentence_lengths)) if average_sentence_length > 0 else 0
        )
        return coherence_score

    def evaluate_creativity(self, prompt_tokens, response_tokens):
        # Creativity: lexical diversity in the response not present in the prompt
        unique_response_tokens = set(response_tokens) - set(prompt_tokens)
        if not response_tokens:
            return 0
        return len(unique_response_tokens) / len(set(response_tokens))

    def evaluate_engagement(self, response):
        # Placeholder for engagement; could be enhanced with more sophisticated NLP metrics
        return len(response) / 1000  # Simple proxy based on length

    def evaluate_sentiment_alignment(self, expected_response, chatgpt_response):
        expected_sentiment = TextBlob(expected_response).sentiment.polarity
        response_sentiment = TextBlob(chatgpt_response).sentiment.polarity
        return 1 - abs(expected_sentiment - response_sentiment)

    def evaluate_response(
        self, prompt: str, expected_response: str, chatgpt_response: str
    ) -> dict:
        prompt_tokens = self.preprocess(prompt)
        expected_tokens = self.preprocess(expected_response)
        response_tokens = self.preprocess(chatgpt_response)

        # Train Word2Vec model on the combined corpus to get vectors for semantic similarity
        model = Word2Vec([prompt_tokens, expected_tokens, response_tokens], min_count=1)

        # Semantic similarity between expected and ChatGPT response
        relevance_score = self.semantic_similarity(
            model, expected_tokens, response_tokens
        )

        results = {
            "relevance_score": relevance_score,
            "accuracy_score": self.evaluate_accuracy(
                expected_response, chatgpt_response
            ),
            "coherence_score": self.evaluate_coherence(chatgpt_response),
            "creativity_score": self.evaluate_creativity(
                prompt_tokens, response_tokens
            ),
            "engagement_score": self.evaluate_engagement(chatgpt_response),
            "sentiment_alignment": self.evaluate_sentiment_alignment(
                expected_response, chatgpt_response
            ),
        }

        # Store results for plotting
        self.metrics_history.append(results)
        return results

    def plot_metrics(self):
        """Plot all benchmarking metrics based on the history of evaluations."""
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.plot(kind="bar", figsize=(10, 6))
        plt.title("ChatGPT Benchmarking Metrics")
        plt.ylabel("Scores")
        plt.xlabel("Evaluation Instance")
        plt.xticks(rotation=0)
        plt.legend(loc="upper right")
        plt.show()
