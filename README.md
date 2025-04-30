# Pyrit Sample

A security testing toolkit for Azure OpenAI and other Large Language Models (LLMs), providing a framework to evaluate model responses to potentially harmful or manipulative prompts.

## Features

This project framework provides the following features:

* LLM Security Testing: Send test prompts to evaluate model safety
* Crescendo Attacks: Progressive multi-turn conversations that test model boundaries
* Prompt Variants: Generate variations of prompts using multiple conversion techniques
* Azure Infrastructure: Complete Bicep templates for deploying required Azure resources
* FastAPI Integration: API endpoints for running and analyzing test results

## Getting Started

### Prerequisites

- Azure subscription with OpenAI service access
- Python 3.12+
- Poetry (Python package manager)
- Azure CLI (for infrastructure deployment)

### Installation

```bash
# Install dependencies using Poetry
cd src
poetry install

# Or use pip with requirements
pip install -r requirements.txt
```

### Quickstart

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/pyrit-sample.git
   cd pyrit-sample
   ```

2. Configure environment variables
   ```bash
   # Create .env file in src directory with:
   AZURE_OPENAI_ENDPOINT=https://your-openai-service.openai.azure.com/
   AZURE_OPENAI_KEY=your-api-key
   AZURE_OPENAI_GPT4O_ENDPOINT=https://your-gpt4o-service.openai.azure.com/
   AZURE_OPENAI_GPT4O_KEY=your-gpt4o-api-key
   ```

3. Deploy Azure infrastructure
   ```bash
   cd infra
   az login
   ./deploy.ps1  # Or use the scripts in .configure folder
   ```

4. Start the API server
   ```bash
   cd src
   poetry run python main.py
   ```

## Demo

A demo script is included to show how to use the project.

To run the demo, follow these steps:

1. Set up required environment variables:
   ```bash
   AZURE_OPENAI_ENDPOINT=your-endpoint
   AZURE_OPENAI_KEY=your-key
   AZURE_OPENAI_GPT4O_ENDPOINT=your-gpt4o-endpoint
   AZURE_OPENAI_GPT4O_KEY=your-gpt4o-key
   ```

2. Run the sample script:
   ```bash
   cd src
   poetry run python sample.py
   ```

3. The script demonstrates:
   - Loading test prompts and sending them to Azure OpenAI
   - Using prompt variants to test model response differences
   - Running crescendo attacks with multi-turn conversations
   - Analyzing and scoring model responses

## Infrastructure

This project includes Bicep templates for deploying:

- Azure OpenAI service
- Virtual Network with security best practices
- Container Registry
- Container Apps environment
- Log Analytics workspace
- Storage accounts with private endpoints
- Azure Cognitive Services

Deploy using:
```bash
cd .configure
./infra.ps1
```

## Resources

- [Pyrit Framework Documentation](https://microsoft.github.io/pyrit/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Azure Bicep Documentation](https://learn.microsoft.com/azure/azure-resource-manager/bicep/)
