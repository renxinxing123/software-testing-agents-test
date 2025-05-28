# Software Testing Agents with LangChain, CAMEL-AI, and Crew

This project demonstrates a **multi-agent software testing system** built using Coral Protocol, supporting agents from three different frameworks—LangChain, CAMEL-AI, and Crew. The system enables automatic understanding of codebases, pull request testing, and test coverage analysis in any compatible GitHub repository.

---

## ✨ Key Features

This project currently supports **three main functionalities**:

1. **Comprehensive Repository Understanding**
   Automatically analyzes a GitHub repository and provides high-level summaries and usage instructions.

2. **Unit Test Execution for New PRs**
   Automatically runs unit tests related to code changes in new pull requests.

3. **Unit Test Coverage Evaluation for New PRs**
   Reviews code changes in a pull request and evaluates whether all necessary cases are covered by existing unit tests, suggesting improvements if needed.

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

* **RepoUnderstandingAgent (LangChain):**
  Analyzes the entire repository, providing comprehensive summaries and usage instructions.

* **RepoUnitTestAdvisorAgent (LangChain):**
  Assesses whether new PRs are sufficiently covered by existing unit tests, and recommends additional tests if necessary.

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

# Terminal 5: RepoUnderstanding Agent
python 4-langchain-RepoUnderstandingAgent.py

# Terminal 6: RepoUnitTestAdvisor Agent
python 5-langchain-RepoUnitTestAdvisorAgent.py
```

---

## Usage Examples

### 1. **Repository Understanding**

Ask the Interface Agent for a comprehensive summary of a repository:

```
Please give me a comprehensive instruction of Coral-Protocol/coraliser.
```

Or specify a branch:

```
Please give me a comprehensive instruction of the master branch of Coral-Protocol/coral-server.
```

Or for other public repositories:

```
Please give me a comprehensive instruction of the master branch of camel-ai/camel.
```

---

### 2. **Unit Test Execution for New PRs**

Ask the system to execute all relevant unit tests for a PR:

```
Please execute the unit test for the '6' PR in repo 'renxinxing123/software-testing-code'.
Please execute the unit test for the '2' PR in repo 'renxinxing123/camel-software-testing'.
```

---

### 3. **Unit Test Coverage Evaluation**

Ask the system to evaluate whether a PR’s changes are fully covered by tests:

```
I created a new branch, `new-semantic-scholar-toolkit`, in the repository `renxinxing123/camel-software-testing` and opened a new pull request (#3). For the changed files, could you please help me check whether the corresponding unit tests fully cover all necessary cases? Are there any additional tests that should be added?
```

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

