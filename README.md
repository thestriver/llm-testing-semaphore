# LLM Performance Testing with LangChain

## Introduction
This repository contains Python code for performance testing Large Language Models (LLMs) using LangChain Evaluators. It includes pytest test files for evaluating LLMs against criteria such as conciseness and correctness.

## Setup
To set up the testing environment, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/thestriver/llm-testing-semaphore
    ```
2. Navigate to the repository directory:
    ```sh
    cd your-repo-name
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
4. Set your OpenAI API key as an environment variable (for local testing):
    ```sh
    export OPENAI_API_KEY='your_openai_api_key_here'
    ```

## Running Tests
To run the tests locally, execute the following command:

```sh
pytest
```