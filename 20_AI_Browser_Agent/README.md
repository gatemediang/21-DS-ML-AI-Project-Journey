# AI Browser Agent for Web Automation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gemini AI](https://img.shields.io/badge/AI-Google%20Gemini-4285F4)](https://ai.google.dev/)

## 📋 Overview

An intelligent browser automation agent powered by Google's Gemini AI that can autonomously navigate the web, search for information, and extract structured data. This project demonstrates how to combine Large Language Models (LLMs) with browser automation to create agents capable of performing complex web tasks with natural language instructions.

## 🎯 The Task

**Objective**: Build an AI-powered browser agent that can:
- Accept natural language task descriptions
- Autonomously navigate web browsers
- Search and extract information from websites
- Structure and save results in JSON format

**Use Case Example**: 
> "Search Google for 'what is browser automation' and summarize the top 3 results in a JSON file."

The agent intelligently interprets this instruction, performs the search, analyzes results, and outputs structured data.

## 🎥 Video Demonstration

Watch the complete walkthrough and see the results:
- **Demo Video**: [https://www.loom.com/share/f7d297cddcc44d29b57d62abf518e53b](https://www.loom.com/share/f7d297cddcc44d29b57d62abf518e53b)

## 🔧 Solutions & Architecture

### Problem Statement
Traditional web scraping and automation tools require:
- Manual script writing for each task
- Brittle selectors that break with website changes
- No contextual understanding of content
- Complex logic for decision-making

### Our Solution
This project leverages **AI-driven browser automation** to solve these challenges:

1. **Natural Language Interface**: Users describe tasks in plain English
2. **Intelligent Navigation**: AI understands context and makes decisions
3. **Adaptive Scraping**: No hardcoded selectors needed
4. **Structured Output**: Automatic JSON formatting of results

### Technology Stack

#### 1. **Browser-Use Library**
- **Purpose**: Provides the browser automation framework with AI integration
- **Why**: Purpose-built for LLM-driven browser control, simplifying complex automation tasks
- **Features**: Async support, session management, visual feedback

#### 2. **Google Gemini 2.5 Flash**
- **Purpose**: Powers the intelligent decision-making and content analysis
- **Why**: 
  - Fast inference for real-time browser control
  - Excellent reasoning capabilities for web navigation
  - Native multimodal support (text + screenshots)
  - Cost-effective for automation tasks

#### 3. **Python Asyncio**
- **Purpose**: Enables asynchronous browser operations
- **Why**: Non-blocking execution allows the agent to handle multiple tasks efficiently

#### 4. **Python-dotenv**
- **Purpose**: Secure environment variable management
- **Why**: Keeps API keys and sensitive data out of source code

## 🚀 Implementation Guide

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))
- Windows/macOS/Linux operating system

### Step 1: Clone the Repository

```bash
git clone <[text](https://github.com/gatemediang/21-DS-ML-AI-Project-Journey.git)>
cd 20_AI_Browser_Agent
```

### Step 2: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv my_env

# Activate virtual environment
# On Windows:
.\my_env\Scripts\Activate.ps1

# On macOS/Linux:
source my_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install browser-use python-dotenv google-generativeai

# Install Playwright browsers
playwright install chromium --with-deps
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root directory:

```bash
# Copy example environment file
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

**Security Note**: The `.env` file is already included in `.gitignore` to prevent accidental exposure of API keys.

### Step 5: Run the Agent

```bash
python browser.py
```

### Step 6: Customize Tasks

Edit the `task` variable in `browser.py`:

```python
# Example tasks:
task = "Find the top 5 trending repositories on GitHub"
task = "Search Hacker News for AI articles and summarize the top 3"
task = "Navigate to Amazon and find laptops under $1000"
```

## 📊 Output Format

The agent saves results to `output.json` in the following structure:

```json
[
  {
    "title": "Article Title",
    "summary": "AI-generated summary of the content"
  },
  {
    "title": "Another Article",
    "summary": "Summary of this article..."
  }
]
```

## 📁 Project Structure

```
20_AI_Browser_Agent/
├── browser.py          # Main agent script
├── output.json         # Generated results
├── .env               # Environment variables (not in git)
├── .env.example       # Template for environment setup
├── .gitignore         # Git ignore rules
├── README.md          # This file
└── my_env/            # Virtual environment (not in git)
```

## 🔑 Key Features

- ✅ **Natural Language Control**: Describe tasks in plain English
- ✅ **AI-Powered Navigation**: Intelligent decision-making for complex workflows
- ✅ **Automatic Data Extraction**: No manual selector writing
- ✅ **JSON Output**: Structured, parseable results
- ✅ **Async Architecture**: Efficient, non-blocking execution
- ✅ **Secure Configuration**: Environment-based API key management

## 🛠️ Advanced Usage

### Monitoring Browser Activity

The browser-use library provides visual feedback during execution. You can watch the agent navigate in real-time.

### Error Handling

The agent includes built-in error handling for:
- Network timeouts
- Missing elements
- API rate limits
- Invalid task descriptions

### Performance Optimization

For faster execution:
```python
# Use headless mode (no visual browser)
agent = Agent(task=task, llm=llm, headless=True)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025


## 🔗 Resources

- [Browser-Use Documentation](https://github.com/browser-use/browser-use)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Playwright Documentation](https://playwright.dev/)
- [Project Demo Video](https://www.loom.com/share/f7d297cddcc44d29b57d62abf518e53b)

## 💡 Use Cases

- **Research Automation**: Gather information from multiple sources
- **Content Monitoring**: Track changes on websites
- **Data Collection**: Extract structured data from web pages
- **Competitive Analysis**: Monitor competitor websites
- **News Aggregation**: Collect and summarize news articles
- **Price Monitoring**: Track product prices across e-commerce sites

## 🐛 Troubleshooting

### Common Issues

**API Key Not Found**
```bash
# Verify .env file exists and contains GEMINI_API_KEY
cat .env
```

**Playwright Installation Failed**
```bash
# Install with system dependencies
playwright install chromium --with-deps
```

**Import Errors**
```bash
# Reinstall dependencies
pip install --upgrade browser-use python-dotenv google-generativeai
```

## 📞 Support

For questions or issues, please:
1. Check the [video demonstration](https://www.loom.com/share/f7d297cddcc44d29b57d62abf518e53b)
2. Review existing GitHub issues
3. Open a new issue with detailed information

---

**Built with ❤️ using Google Gemini AI and Browser-Use**