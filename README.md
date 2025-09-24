Description ✨ This project is an AI Chatbot. It uses a virtual environment to manage dependencies, connects to the OpenAI API for its core functionality, and is built using Streamlit to provide an interactive user interface. The project setup includes a .gitignore file to protect sensitive information like API keys, a .env file for environment variables, and a standard main.py file for the application's code.
Features 🤖 Interactive Chat Interface: A user-friendly interface built with Streamlit.
OpenAI Integration: Connects to the OpenAI API to generate responses.
Dependency Management: Uses a virtual environment to keep project dependencies organized and isolated.
Secure API Key Handling: Utilizes a .env file to securely store the API key, which is ignored by Git.
Installation 🔧 Follow these steps to set up and run the project locally.
Clone the Repository:
Bash
git clone cd Set Up the Virtual Environment:
macOS:
Bash
pyenv local 3.8 python3 -m venv venv source venv/bin/activate Windows:
Bash
pyenv local 3.8 python -m venv venv .venv\Scripts\Activate.ps1 Install Dependencies:
Bash
pip install -r requirements.txt Note: You may need to create a requirements.txt file by running pip freeze > requirements.txt after installing all packages.
Configure Environment Variables:
Rename the .env.example file to .env.
Open the .env file and add your OpenAI API key:
OPENAI_API_KEY=your_api_key_here
Usage ⌨️ To run the application, make sure your virtual environment is activated, and then execute the following command:
Bash
streamlit run main.py
 
The app will open in your default web browser.
Project Structure 📁 Here's a brief overview of the key files in the project:
.env: Stores environment variables like the OpenAI API key. (Not tracked by Git)
.env.example: A template for the .env file.
.gitignore: Specifies files and directories to be ignored by Git.
main.py: The main script for the Streamlit application.
venv/: The project's virtual environment directory.


 
