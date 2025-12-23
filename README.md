# Drive RAG Assistant

This application allows you to chat with your own documents (PDFs, Word docs,
Text files, etc.) stored on your computer. It uses Artificial Intelligence to
read your files and answer your questions based on their content.

## Features

- **Chat with your data**: Ask questions and get answers cited from your local
  files.
- **Privacy-focused**: Your documents stay on your computer.
- **Flexible AI Options**: choice of OpenAI, Google, Anthropic, or completely
  free/local AI with Ollama.
- **User-Friendly Interface**: Simple web-based chat interface.

## Sample Data

The project includes a `test_DB` folder containing synthetic employee data
generated using **Gemini 2.5 Flash**. You can use this folder to test the
application immediately without needing your own documents.

## Application Structure

- `app.py`: The main application file that runs the user interface.
- `ingest.py`: Handles reading your files and converting them into a format the
  AI can understand ("vectorizing").
- `answer.py`: The logic for finding the right documents and generating an
  answer.
- `model_config.py`: Manages connections to different AI providers (OpenAI,
  Ollama, etc.).

## Prerequisites

Before running the application, you need:

1. **Python**: Ensure you have Python installed (version 3.10 or higher is
   recommended).
2. **Ollama (Optional)**: If you want to use the app for free without internet
   dependencies, install [Ollama](https://ollama.com/) and download the Llama
   3.2 model.

### Setting up Ollama (For Free/Local Use)

1. Download and install Ollama from [ollama.com](https://ollama.com).
   Alternatively, if you are on Mac, you can install it via Homebrew:
   `brew install ollama`.
2. Open your terminal or command prompt.
3. Run the following command to download the model:
   ```bash
   ollama pull llama3.2:latest
   ```
4. Keep the Ollama application running in the background while using this tool.

## Installation

1. **Open Terminal**: Open your command prompt or terminal.
2. **Navigate to the folder**: Use `cd` to go to the project directory.
   ```bash
   cd path/to/drive-rag-assistant
   ```
3. **Install Requirements**: Run the helper command to install all necessary
   libraries.
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Start the application by running:
   ```bash
   python app.py
   ```
2. Wait for a message that says `Running on local URL: http://127.0.0.1:7860`.
3. Open that link in your web browser.

## How to Use

1. **Select Provider**: On the left sidebar, choose your AI provider.
   - **ollama(free)**: Select this if you followed the Ollama setup above. No
     API key needed.
   - **openai / google / anthropic**: Select one of these if you have a paid API
     key. Enter your key in the "API key" box.
2. **Select Database**: Click the file explorer in the sidebar to select the
   folder containing your documents. **Important**: To select a folder, open it
   and then click the folder icon with a `.` (dot) next to it.
3. **Vectorize**: Click the "Vectorize The Database" button. This reads your
   files and prepares them. You only need to do this once, or whenever you add
   new files.
4. **Chat**: Type your question in the main chat window and press Enter.

## Troubleshooting

- **"NameError: ChatOllama is not defined"**: Ensure you have installed the
  requirements correctly.
- **Ollama not working**: Make sure the Ollama app is running and you have
  pulled the `llama3.2:latest` model.
- **Dependencies errors**: If you see errors about missing modules, try running
  `pip install -r requirements.txt` again.
