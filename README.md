
# Thesis on LLMs for Product Conception Efficiency

This repository is dedicated to the research conducted for a Master's thesis on the use of Large Language Models (LLMs) to enhance efficiency in the product conception process. The study compares traditional human-driven product conception against product conceptions developed with the guidance of ChatGPT and Mixtral, aiming to evaluate the potential of LLMs in streamlining and innovating the conceptual phase of product development.

## Repository Structure

- `benchmarking/`: Contains the core benchmarking classes and documentation necessary for evaluating the efficiency and creativity of product conceptions generated by humans, ChatGPT, and Mixtral.
  - `classes/`: Includes Python classes for benchmarking the performance of ChatGPT (`GPTBenchmark.py`) and Mixtral (`MixtralBenchmark.py`).
  - `docs/`: Documentation on the methodologies used for benchmarking, including the advantages of centroid plotting and detailed explanations of the benchmarking process.
- `prompt_creation/`: Scripts for generating prompts used in the product conception process with ChatGPT and Mixtral. This includes a direct approach for ChatGPT and the use of the llama-index for Mixtral.
- `main.py`: The primary script that orchestrates the benchmarking process, guiding the user through the evaluation of product conception methods.

## Getting Started

### Prerequisites

- Python 3.9 or newer.
- Installation of necessary Python libraries as listed in `requirements.txt`.

### Installation

To set up the project for use:

1. Clone this repository:
   ```bash
   git clone https://github.com/gmatasruiz/llm_product_inception.git
   ```
2. Navigate into the cloned directory and install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the evaluation process, run `main.py` and follow the prompts to conduct benchmarking across different product conception methods:

```bash
python main.py
```

## Benchmarking Overview

The benchmarking framework is built upon sophisticated NLP and statistical methodologies to assess the efficiency and creativity of product conceptions. The core components and their roles in enhancing the benchmarking process include:

- **Vectorization and Semantic Analysis**:
    - Utilizes TF-IDF vectorization to transform textual data into numerical vectors, enabling the quantitative analysis of text. This process is critical for comparing the semantic content of prompts and generated responses.
    - Employs Principal Component Analysis (PCA) for dimensionality reduction, facilitating the visualization of prompts in a 2D space. This visualization aids in identifying thematic concentrations and diversity within the prompts.

- **Semantic Similarity Measurement**:
    - Implements Word2Vec to compute high-dimensional vector representations of words, capturing the semantic relationships between them. This model is used to calculate the semantic similarity between expected responses and those generated by LLMs, providing a metric for relevance and contextual accuracy.

- **Comprehensive Evaluation Metrics**:
    - **Accuracy**: Assessed heuristically based on the response length relative to the expected response. Future implementations could include more sophisticated domain-specific accuracy assessments.
    - **Coherence**: Evaluated through readability metrics, specifically analyzing sentence length and complexity. This metric offers insight into the logical structure and fluency of the generated text.
    - **Creativity**: Measured by the lexical diversity within responses, particularly focusing on the uniqueness of the content generated in response to prompts.
    - **Engagement**: A proxy measure based on the length and structural complexity of responses, indicating the potential to captivate the target audience.
    - **Sentiment Alignment**: Analyzes the emotional tone of responses compared to expected outcomes, using sentiment analysis to ensure alignment with the intended sentiment of prompts.

### Plotting for Insights
- **Centroid Plotting**: Through PCA, centroids of vectorized prompts are plotted, providing a graphical representation of thematic distributions and identifying areas for prompt diversification.
- **Metric Visualization**: The suite plots the history of evaluation metrics across multiple instances, offering a visual summary of performance trends and highlighting areas for improvement.

This benchmarking framework not only scrutinizes the current capabilities of LLMs in product conception but also guides the iterative refinement of prompts and model parameters to enhance future performance.


## Contributing

This research project welcomes contributions, especially in the form of additional benchmarking methodologies, prompt enhancement techniques, or analysis scripts. See `CONTRIBUTING.md` for guidelines on making contributions.

## License

This project is released under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- The developers and community behind ChatGPT and Mixtral for providing the tools necessary for this research.
- Faculty advisors and peers who have offered invaluable feedback and support.
