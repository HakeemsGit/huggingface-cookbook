"""
LLM-as-a-judge for automated and versatile evaluation

This script implements an automated evaluation system using LLMs as judges,
as introduced in "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena".

The idea is to use an LLM to grade other LLMs' outputs, providing a scalable
way to evaluate broad capabilities without requiring human evaluation time.
"""

import re
import sys
import psutil
import pandas as pd
from tqdm.auto import tqdm
from llama_cpp import Llama

# Safety check function
def check_system_resources():
    """Check if system has enough resources"""
    min_memory = 4 * 1024 * 1024 * 1024  # 4GB
    available = psutil.virtual_memory().available
    if available < min_memory:
        raise SystemError(f"Not enough available memory to run safely. Need {min_memory/1e9:.1f}GB, have {available/1e9:.1f}GB")
    print(f"Available memory: {available/1e9:.1f}GB")

# Enable tqdm for pandas operations
tqdm.pandas()
pd.set_option("display.max_colwidth", None)

# Initialize LLM globally
LLM = None

BASIC_JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer as a float on a scale of 0 to 10, where 0 means that the system_answer is not helpful at all, and 10 means that the answer completely and helpfully addresses the question.

Provide your feedback as follows:

Feedback:::
Total rating: (your rating, as a float between 0 and 10)

Now here are the question and answer.

Question: {question}
Answer: {answer}

Feedback:::
Total rating: """

IMPROVED_JUDGE_PROMPT = """
You will be given a user_question and system_answer couple.
Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.

Here is the scale you should use to build your answer:
1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
2: The system_answer is mostly not helpful: misses some key aspects of the question
3: The system_answer is mostly helpful: provides support, but still could be improved
4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question

Provide your feedback as follows:

Feedback:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 4)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and answer.

Question: {question}
Answer: {answer}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Feedback:::
Evaluation: """

def setup_llm_client(model_path="models/llama-2-7b-chat.gguf"):
    """Initialize the LLM client for evaluation"""
    global LLM
    if LLM is None:
        LLM = Llama(
            model_path=model_path,
            n_ctx=734,      # Context window
            n_threads=2,    # Reduced CPU threads
            n_batch=1,      # Minimal batch size
            low_vram=True,  # Enable low VRAM mode
            seed=42,        # Set seed for reproducibility
            verbose=False   # Reduce logging
        )
    return LLM

def extract_judge_score(answer: str, split_str: str = "Total rating:") -> float:
    """Extract numerical score from LLM judge output"""
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception as e:
        print(e)
        return None

def load_evaluation_dataset():
    """Load and prepare the FeedbackQA dataset from local files"""
    # Load the local JSON file
    ratings = pd.read_json("data/feedback_train.json")
    
    # Extract the answer from the passage content
    ratings["answer"] = ratings["passage"].apply(lambda x: x["reference"]["section_content"])
    
    # Extract ratings and explanations from the feedback lists
    ratings["review_1"] = ratings["rating"].apply(lambda x: x[0])
    ratings["explanation_1"] = ratings["feedback"].apply(lambda x: x[0])
    ratings["review_2"] = ratings["rating"].apply(lambda x: x[1])
    ratings["explanation_2"] = ratings["feedback"].apply(lambda x: x[1])
    ratings = ratings.drop(columns=["feedback", "rating", "passage", "domain"])

    # Map scores to numeric values
    conversion_dict = {"Excellent": 4, "Acceptable": 3, "Could be Improved": 2, "Bad": 1}
    ratings["score_1"] = ratings["review_1"].map(conversion_dict)
    ratings["score_2"] = ratings["review_2"].map(conversion_dict)
    
    return ratings

def evaluate_with_llm(examples, llm_client, prompt_template):
    """Run LLM evaluation on the examples"""
    results = []
    
    # Process one example at a time to minimize memory usage
    for idx, row in tqdm(examples.iterrows(), total=len(examples)):
        try:
            result = llm_client.create_completion(
                prompt=prompt_template.format(question=row["question"], answer=row["answer"]),
                max_tokens=250,  # Reduced max tokens
                temperature=0.1,
                stop=["Question:", "Feedback:::"]
            )["choices"][0]["text"]
            results.append(result)
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            results.append(None)
            
    examples["llm_judge"] = results
    examples["llm_judge_score"] = examples["llm_judge"].apply(extract_judge_score)
    
    if prompt_template == BASIC_JUDGE_PROMPT:
        # Rescale basic prompt scores to 1-4 range
        examples["llm_judge_score"] = (examples["llm_judge_score"] / 10) + 1
        
    return examples

def main():
    """Main execution flow"""
    # Check system resources
    check_system_resources()
    
    try:
        # Setup
        llm_client = setup_llm_client()
        ratings = load_evaluation_dataset()
        
        # Get examples where raters agree but limit sample size
        ratings_where_raters_agree = ratings.loc[ratings["score_1"] == ratings["score_2"]]
        examples = ratings_where_raters_agree.groupby("score_1").sample(2, random_state=1214)  # Reduced samples
    except Exception as e:
        print(e)

    examples["human_score"] = examples["score_1"]

    # Calculate human rater correlation baseline
    human_correlation = ratings["score_1"].corr(ratings["score_2"], method="pearson")
    print(f"Correlation between human raters: {human_correlation:.3f}")

    # Evaluate with basic prompt
    examples = evaluate_with_llm(examples, llm_client, BASIC_JUDGE_PROMPT)
    basic_correlation = examples["llm_judge_score"].corr(examples["human_score"], method="pearson")
    print(f"Basic LLM judge correlation: {basic_correlation:.3f}")

    # Evaluate with improved prompt
    examples = evaluate_with_llm(examples, llm_client, IMPROVED_JUDGE_PROMPT)
    improved_correlation = examples["llm_judge_score"].corr(examples["human_score"], method="pearson")
    print(f"Improved LLM judge correlation: {improved_correlation:.3f}")

if __name__ == "__main__":
    main()
