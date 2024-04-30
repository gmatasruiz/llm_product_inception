from pprint import pprint
from classes.SpecificLLMBenchmark import ChatGPTBenchmark

# Example usage
benchmark = ChatGPTBenchmark()

examples = {
    "case_1": {
        "title": "High Similarity and Coherence",
        "prompt": "How can AI improve the efficiency of remote work environments?",
        "expected_response": """AI can enhance remote work environments by automating routine tasks, 
          providing real-time collaboration tools, and personalizing learning and development programs. 
          These improvements can increase productivity and reduce the time required for administrative tasks.""",
        "chatgpt_response": """AI technologies can significantly improve remote work by automating mundane tasks,
          offering advanced tools for real-time collaboration, and tailoring learning experiences to individual needs.
          This not only boosts efficiency but also saves valuable time that can be better spent on more complex projects.""",
    },
    "case_2": {
        "title": "Very Different Sentiment",
        "prompt": "What are the potential risks of implementing AI in healthcare?",
        "expected_response": """Implementing AI in healthcare can lead to privacy concerns, as sensitive patient data might be 
        exposed. There's also a risk of over-reliance on technology which might lead to errors if the AI fails or malfunctions.""",
        "chatgpt_response": """AI in healthcare presents exciting opportunities to enhance patient care and streamline operations. 
        While there are challenges, such as integrating with existing systems, the benefits, including faster diagnosis and treatment 
        options, can significantly outweigh the risks.""",
    },
}

for k in examples.keys():
    pprint(f"Case: {examples[k]['title']}")
    evaluation = benchmark.evaluate_response(
        examples[k]["prompt"],
        examples[k]["expected_response"],
        examples[k]["chatgpt_response"],
    )
    pprint(evaluation)
