# ChatMate: AI Assistant with Task Management

A modern AI assistant application with task management capabilities, powered by Ollama and cloud AI services.

## Features

- Advanced AI assistant with multiple model support (local and cloud-based)
- WhatsApp-style chat interface with message bubbles
- Image upload and analysis capabilities
- Task management with categories (To Do, Ongoing, Done, Waiting, Someday)
- Dark theme with modern UI inspired by messaging apps
- Chat history storage and persistence
- Resizable interface
- Offline mode with fallback responses

## Requirements

- Python 3.6+
- PyQt5
- Ollama (running locally with Gemma2 model)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure Ollama is installed and running with the llama3 model:

```bash
# Install Ollama from https://ollama.ai/
# Make sure you have the Gemma2 model
ollama pull gemma2
```

3. Run the application:

```bash
python main.py
```

## Usage

### Tasks Tab

- Add new tasks using the input field at the top
- View tasks by category using the tabs
- Change a task's status using the dropdown menu
- Delete tasks with the × button

### ChatMate Tab

- Chat with the AI assistant powered by Ollama or cloud AI services
- Upload and analyze images using the image button
- Switch between different AI models using the dropdown
- Configure online AI services with the cloud button
- Ask for help with task management, productivity tips, or any other questions

## Note

The application saves your todos to a local file (`todos.json`) automatically.
