from benchmarking.classes.SpecificLLMBenchmark import ChatGPTBenchmark

# Example usage
benchmark = ChatGPTBenchmark()
prompt = "Tell me a story about a lonely robot."
expected_response = "Once upon a time, in a world not unlike our own, there was a lonely robot named Rob. Rob had been designed to interact with humans..."
chatgpt_response = "There was once a robot named Solo, who wandered the earth without a friend. Its circuits yearned for companionship, but it found solace in the beauty of the natural world..."

evaluation = benchmark.evaluate_response(prompt, expected_response, chatgpt_response)

benchmark.plot_metrics()
print(evaluation)