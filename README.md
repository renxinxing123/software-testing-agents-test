# Software Testing Agents with LangChain, CAMEL-AI, and Crew

This project demonstrates a **multi-agent software testing system** built using Coral Protocol, supporting agents from three different frameworks—LangChain, CAMEL-AI, and Crew. The system enables automatic understanding of codebases, pull request testing, and test coverage analysis in any compatible GitHub repository.

---

## ✨ Key Features

This project currently supports **three main functionalities**:

1. **Unit Test Execution for New PRs**
   Automatically runs unit tests related to code changes in new pull requests.

---

## Overview of Agents

The system consists of six cooperating agents, each with a specific responsibility:

* **Interface Agent (LangChain):**
  Accepts user instructions, manages workflow, and coordinates other agents.

* **GitCloneAgent (Crew):**
  Clones the GitHub repository and checks out the specified pull request branch.

* **CodeDiffReviewAgent (CAMEL-AI):**
  Analyzes the PR diff, identifies changed functions, maps to corresponding tests, and locates relevant test files.
  **Tip:**

  * By default, the CodeDiffReviewAgent uses OpenAI GPT-4.1 for analysis.
  * To try **Groq Llama 3 70B** for improved cost and speed, you can comment out the following code:

    ```python
    # model = ModelFactory.create(
    #     model_platform=ModelPlatformType.OPENAI,
    #     model_type=ModelType.GPT_4_1,
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     model_config_dict={"temperature": 0.3, "max_tokens": 32147},
    # )
    ```

    and **use this code instead**:

    ```python
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GROQ,
        model_type=ModelType.GROQ_LLAMA_3_3_70B,
        model_config_dict={"temperature": 0.3},
    )
    ```

    > Note: Swapping out other models may currently affect system performance or coverage accuracy.

* **UnitTestRunnerAgent (LangChain):**
  Runs specified unit tests using `pytest` and returns structured results.

---

## Prerequisites

* Clone the [Coral MCP server](https://github.com/Coral-Protocol/coral-server):

  ```bash
  git clone https://github.com/Coral-Protocol/coral-server.git
  ```
* Python 3.10 or above
* Export a valid `OPENAI_API_KEY` and `GITHUB_ACCESS_TOKEN` in your environment

---

## Installation

```bash
pip install PyGithub
pip install langchain-mcp-adapters langchain-openai langchain langchain-core
pip install crewai
pip install 'camel-ai[all]'
```

---

## Getting Started

### 1. Start the MCP Server

Navigate to the `coral-server` directory and run:

```bash
./gradlew run
```

> Note: Gradle may appear to stall at 83%, but the server is running. Check terminal logs to confirm.

---

### 2. Launch Agents (in six separate terminals)

```bash
# Terminal 1: Interface Agent
python 0-langchain-interface.py

# Terminal 2: GitClone Agent
python 1-crewai-GitCloneAgent.py

# Terminal 3: CodeDiffReview Agent
python 2-camel-CodeDiffReviewAgent.py

# Terminal 4: UnitTestRunner Agent
python 3-langchain-UnitTestRunnerAgent.py
```

---

## Usage Examples

### 1. **Unit Test Execution for New PRs**

Ask the system to execute all relevant unit tests for a PR:

```
Please execute the unit test for the '6' PR in repo 'renxinxing123/software-testing-code'.
Please execute the unit test for the '2' PR in repo 'renxinxing123/camel-software-testing'.
```

**🎬 [Watch Video Demo](https://youtu.be/-ZYZEo96L1w)**

---

## Notes

* When running tests, the system identifies the relevant unit test files and executes all test cases within them for reliable coverage.
* The project is designed for easy extension—feel free to add new agent scripts or tools!
* **Switching CodeDiffReviewAgent to Groq Llama 3 70B** may help reduce cost and increase speed, but the quality of system coverage might be slightly affected compared to GPT-4.1.

---

## Get Involved

This is an early-stage prototype. Feedback and contributions are welcome!

Discord: [https://discord.gg/cDzGHnzkwD](https://discord.gg/cDzGHnzkwD)

---

