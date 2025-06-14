# Econometrics AI Agent

This repository hosts the official implementation for the research paper: "[Can AI Master Econometrics? Evidence from Econometrics AI Agent on Expert-Level Tasks](https://arxiv.org/abs/2506.00856)".

The **Econometrics AI Agent** is an LLM-driven, specialized AI agent designed to automate complex econometric analysis, traditionally requiring significant human expertise. This project addresses the challenge of applying AI to intricate, domain-specific tasks where general-purpose Large Language Models (LLMs) and AI agents often fall short without costly fine-tuning or specialized retraining.

Built upon the open-source **MetaGPT framework**, our agent leverages a **zero-shot learning approach**. This allows for the effective integration of deep econometric knowledge through a specialized **econometric 'tool library'** and meticulously crafted prompts. This design avoids the need for expensive and time-consuming LLM retraining, making advanced econometric methods more accessible.

## Demo Video

Below is a demonstration of the Econometrics AI Agent in action:

https://github.com/user-attachments/assets/30732765-f718-43fb-9c60-f46aaa8e63c8

## Key Capabilities

The Econometrics AI Agent demonstrates robust performance in:

*   **Strategic Planning:** Intelligently planning and decomposing complex econometric tasks into manageable sub-tasks.
*   **Code Generation & Execution:** Automatically generating and executing Python code for various econometric analyses.
*   **Error-Based Reflection:** Employing a reflection mechanism to evaluate action outcomes, learn from errors, and improve the robustness of its solutions.
*   **Iterative Refinement:** Allowing users to engage in multi-round conversations for iterative task refinement and to meet specific user needs.
*   **Domain-Specific Tool Usage:** Utilizing a rich library of pre-defined econometric tools and functions (e.g., IV-2SLS, DID, RDD, Propensity Score Methods) to perform accurate analyses.

## Why Econometrics AI Agent?

Our comparative tests show that this domain-specialized agent significantly **outperforms benchmark LLMs and general-purpose AI agents** (like a general Data Interpreter without the specialized toolkit) in expert-level econometric tasks. This project aims to:

*   **Democratize Expertise:** Make advanced econometric methods accessible to users with minimal coding expertise, including students and practitioners.
*   **Boost Research Productivity:** Serve as a powerful tool for academic researchers and industry practitioners, accelerating the empirical research process.
*   **Enhance Reproducibility:** Contribute to improving the reproducibility of empirical research.
*   **Educational Applications:** Offer promising applications for econometrics teaching and learning.
*   **Cost-Effective Specialization:** Provide a low-cost, high-accuracy solution for injecting domain knowledge into AI systems, bypassing the need for extensive fine-tuning.

This repository provides the complete source code, datasets used for evaluation (from academic coursework and published papers), and the framework for the Econometrics AI Agent. We encourage researchers, students, and practitioners to explore, utilize, and extend its capabilities for their econometric analysis needs.

---

## Installation Guide

Welcome to the Econometrics AI Agent! This guide will help you install and run the project smoothly.

### 1. Clone the Repository

First, clone the Econometrics AI Agent project to your local machine:

```bash
git clone https://github.com/FromCSUZhou/Econometrics-Agent
```

### 2. Navigate to the Project Directory

Change into the project's root directory:

```bash
cd Econometrics-Agent
```

### 3. Set Up Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Open the `.env` file in a text editor. You will need to replace the placeholder values for `OPENAI_API_KEYS` (or other LLM provider keys) and `OPENAI_API_BASE_URLS` with your actual credentials.

Next, configure the model settings specifically for the Econometrics AI Agent. The agent primarily uses the settings defined in `config/config2.yaml`.
Open `config/config2.yaml` and ensure the `model`, `base_url`, and `api_key` under the relevant model provider section (e.g., `openai`, `azure`, `ollama`) are consistent with the values you intend to use (typically matching or derived from your `.env` setup). If you wish to change the underlying Large Language Model or its provider, this is the primary configuration file to modify for the agent's core operations.

For example, if using OpenAI, you would update the `llm` section in `config/config2.yaml`:
```yaml
llm:
  # ... other llm configurations ...
  # Example for OpenAI:
  model: "gpt-4-turbo-preview"  # Your chosen model
  api_type: "openai"            # or "azure", "ollama", etc.
  api_key: "sk-your_openai_api_key_here"
  base_url: "https://api.openai.com/v1" # Your OpenAI API base URL
  # ... other parameters ...
```
Ensure these values reflect your active LLM service subscription.

### 4. Create and Activate a Virtual Environment (using uv)

**First, ensure you have `uv` installed.** If not, you can install it using the following commands:

*   **macOS and Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*   **Windows (PowerShell):**
    ```bash
    irm https://astral.sh/uv/install.ps1 | iex
    ```

**Then, create and activate the virtual environment:**

It is recommended to use Python 3.11. `uv` will try to find a suitable Python version on your system, or you can specify it explicitly:
```bash
uv venv --python 3.11
# For Linux/macOS
source .venv/bin/activate
# For Windows (Command Prompt)
# .venv\Scripts\activate.bat
# For Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### 5. Install Project Dependencies

With the virtual environment activated, install the main project dependencies:

```bash
uv pip install -r requirements.txt
```

### 6. Set Up Frontend Dependencies

Navigate to the `web` directory to install frontend dependencies:

```bash
cd web
npm install
```

### 7. Build Frontend Project

Still in the `web` directory, build the frontend project to generate compiled output files:

```bash
npm run build
```

After the build is complete, navigate back to the project root directory:
```bash
cd ..
```

### 8. Start the Application

Finally, run the start script from the project root directory to launch the application:

```bash
bash start.sh
```

### 9. Access the Application

The application should now be running. You can access it in your web browser at:

[http://localhost:1280](http://localhost:1280)

Enjoy! 😄

---

## Acknowledgements

We extend our sincere gratitude to the developers and communities of the following open-source projects, which provided foundational frameworks and components for the Econometrics AI Agent:

*   **ChatPilot** ([@shibing624/ChatPilot](https://github.com/shibing624/ChatPilot)):
    This project utilized the frontend framework from ChatPilot. We have made the following modifications and enhancements:
    *   a. Implemented user quota management and enhanced file upload capabilities.
    *   b. Developed visualizations for the Econometrics AI Agent, including images, agent workflow diagrams, generated code, and execution results.
    *   c. Improved support for high concurrency, enabling multiple users to utilize the Econometrics AI Agent simultaneously.
    *   d. Addressed various bugs and improved stability.

*   **MetaGPT (Data Interpreter)** ([@FoundationAgents/MetaGPT/tree/code_interpreter](https://github.com/FoundationAgents/MetaGPT/tree/code_interpreter)):
    The backend agent architecture is based on the Data Interpreter from MetaGPT. We adapted and extended it to create the specialized Econometrics AI Agent with the following key improvements:
    *   a. Replaced the original tools and prompts of the Data Interpreter with a specialized set tailored for econometrics tasks.
    *   b. Integrated multi-round conversational capabilities, allowing the Econometrics AI Agent to adjust and inherit task states based on ongoing user interaction.
    *   c. Developed visualizations for the agent's processes and outputs.
    *   d. Resolved various bugs and enhanced overall performance.

---

## Troubleshooting

If you encounter any issues during installation or operation, please consider the following:

- **Network Connectivity:** Ensure your internet connection is stable and can access GitHub, npm repositories, and your LLM provider's API endpoints.
- **Software Versions:** Verify that `uv`, Node.js (>= 18 recommended for frontend development), and Python (version 3.11 recommended) are correctly installed and accessible in your system's PATH.
- **API Keys & Configuration:** Double-check that your API keys and base URLs in `.env` and `config/config2.yaml` are correct, valid, and that the specified model is accessible with your subscription.
- **Dependency Issues:** If you experience problems during `uv pip install` or `npm install`, try clearing the respective caches (`uv pip cache clean` or `npm cache clean --force`) and then attempt reinstallation.
- **Port Conflicts:** Ensure that port `1280` (or the port configured in `start.sh`) is not already in use by another application.
- **Frontend Build:** If the web interface doesn't load correctly, try rebuilding the frontend (steps 6 and 7).
- **Log Files:** Check application logs (which may be printed to the console where `bash start.sh` is running) for any specific error messages.

---
## Citation

If you use the Econometrics AI Agent in your research, please cite our paper:

```bibtex
@misc{chen2025econometricsai,
  title     = {Can AI Master Econometrics? Evidence from Econometrics AI Agent on Expert-Level Tasks},
  author    = {Qiang Chen and Tianyang Han and Jin Li and Ye Luo and Yuxiao Wu and Xiaowei Zhang and Tuo Zhou},
  year      = {2025},
  eprint    = {2506.00856},
  archivePrefix = {arXiv},
  primaryClass  = {econ.EM},
  doi       = {10.48550/arXiv.2506.00856},
  url       = {https://arxiv.org/abs/2506.00856}
}
```

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). See the `LICENSE` file for details.
