import pytest
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain_openai import ChatOpenAI

# Fixture for LLM setup
@pytest.fixture(scope="session")
def chat_openai_llm():
    return ChatOpenAI(model="gpt-4", temperature=0)

# Parametrized tests with expected outcomes and failure handling
@pytest.mark.parametrize("input_text, prediction, reference, criteria, expected_score", [
    # Unlike the basic test, this test case won't fail the conciseness criteria due to verbosity 
    # because we've now placed an expected score of 0.
    ("What's 2+2?", "Of course, that's 4. A simple mathematical fact.", "4", "conciseness", 0),
    # This case is expected to pass the correctness criteria
    ("What is the capital of the US?", "The capital of the US is Washington, D.C.", "Washington, D.C.", "correctness", 1),
    # Add more test cases as needed
])
def test_llm_criteria(chat_openai_llm, input_text, prediction, reference, criteria, expected_score):
    evaluator = LabeledCriteriaEvalChain.from_llm(llm=chat_openai_llm, criteria=criteria)
    llm_eval_result = evaluator.evaluate_strings(prediction=prediction, input=input_text, reference=reference)

    assert llm_eval_result["score"] == expected_score, (
        f"LLM failed to meet the {criteria} criteria. "
        f"Score: {llm_eval_result['score']}. "
        f"Reasoning: {llm_eval_result.get('reasoning', 'No reasoning provided')}"
    )
