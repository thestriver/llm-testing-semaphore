import os
import pytest
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain_openai import ChatOpenAI

# Ensure the OPENAI_API_KEY is set in your environment variables
# assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable not set"

# Initialize the LLM with the OpenAI model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Test for conciseness
# def test_conciseness_criteria():
#     criteria = "conciseness"
#     evaluator = LabeledCriteriaEvalChain.from_llm(
#         llm=llm,
#         criteria=criteria,
#     )
#     llm_eval_result = evaluator.evaluate_strings(
#         prediction="What's 2+2? Of course, that's 4. A simple mathematical fact.",
#         input="What's 2+2?",
#         reference="4",
#     )
#     print(llm_eval_result)
#     assert llm_eval_result["score"] == 1, f"LLM failed to meet the {criteria} criteria"

# Test for correctness
def test_correctness_criteria():
    criteria = "correctness"
    evaluator = LabeledCriteriaEvalChain.from_llm(
        llm=llm,
        criteria=criteria,
    )
    llm_eval_result = evaluator.evaluate_strings(
        prediction="The capital of the US is Washington, D.C.",
        input="What is the capital of the US?",
        reference="Washington, D.C.",
    )
    print(llm_eval_result)
    assert llm_eval_result["score"] == 1, f"LLM failed to meet the {criteria} criteria"
