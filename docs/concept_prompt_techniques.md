# Prompting Techniques Documentation

## Zero-shot Prompting (Template 1)

**Definition**: Zero-shot prompting involves asking the model to perform a task without any prior examples or context.

**How It Works**: The model relies solely on its training data to generate a response based on the prompt.

**Pros**:

- Useful for straightforward tasks
- Quick and easy to implement

**Cons**:

- May produce less accurate results for complex tasks
- Lacks specific guidance

**References**: <https://www.promptingguide.ai/techniques/zeroshot>

## Few-shot Prompting (Template 2)

**Definition**: Few-shot prompting provides a few examples within the prompt to guide the model in generating the desired response.

**How It Works**: This method helps the model understand the context and expected output better by providing relevant examples.

**Pros**:

- Improves accuracy with examples
- Useful for complex tasks

**Cons**:

- Requires careful selection of examples
- May still produce variable results

**References**: <https://www.promptingguide.ai/techniques/fewshot>

## Chain-of-Thought Prompting, CoT (Template 3)

**Definition**: Chain-of-Thought prompting breaks down complex tasks into a series of intermediate steps to facilitate reasoning.

**How It Works**: Combines few-shot examples with step-by-step reasoning to enhance performance on complex tasks.

**Pros**:

- Improves reasoning and accuracy
- Helps with complex tasks

**Cons**:

- More time-consuming
- Requires detailed prompts

**References**: <https://www.promptingguide.ai/techniques/cot>

## Self-Consistency (Template 4)

**Definition**: Self-consistency involves generating multiple reasoning paths for the same prompt and selecting the most consistent answer.

**How It Works**: Improves reliability by considering multiple possible answers and choosing the most consistent one.

**Pros**:

- Enhances reliability
- Reduces variability in answers

**Cons**:

- Computationally expensive
- Time-consuming

**References**: <https://www.promptingguide.ai/techniques/consistency>

## Reflexion (Template 5)

**Definition**: Reflexion uses linguistic feedback from the environment to improve performance.

**How It Works**: Acts as verbal reinforcement learning to help the model learn from prior mistakes.

**Pros**:

- Improves learning
- Enhances performance

**Cons**:

- Requires feedback integration
- Complex to implement

**References**: <https://www.promptingguide.ai/techniques/reflexion>
