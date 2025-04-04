# MCQ_Generator_Using_Langchain_and_HuggingFace

# 🧠 MCQs Creator Application with LangChain & HuggingFace

An intelligent web application built using **LangChain**, **HuggingFace Transformers**, and **Streamlit**, designed to automatically generate multiple-choice questions (MCQs) from any uploaded `.pdf` or `.txt` document.

![App Screenshot](https://mcqgeneratorusinglangchainandhuggingface-ggebgejxgzwg8obpjhfpw.streamlit.app/)

---

## 🚀 Live Demo

👉 [Click here to try the app!](https://mcqgeneratorusinglangchainandhuggingface-ggebgejxgzwg8obpjhfpw.streamlit.app/)  
*Note: Make sure you’re connected to the internet and logged in to Hugging Face if required.*

---

## 📂 Features

- 🔍 **File Upload**: Upload `.pdf` or `.txt` documents.
- 🤖 **AI-Powered MCQ Generation**: Generate MCQs based on subject, complexity, and number of questions.
- 🧪 **Answer Evaluation**: Displays correct answers with choices.
- 📊 **Structured Output**: View MCQs in an interactive table format.
- 🧠 **Review Section**: Shows a short review/commentary generated by the model.

---

## ⚙️ Tech Stack

| Tool/Library      | Purpose                              |
|------------------|--------------------------------------|
| Python            | Core programming language           |
| Streamlit         | Frontend web app                    |
| LangChain         | Prompt chaining and model orchestration |
| HuggingFace       | Language model inference            |
| PyPDF2            | PDF text extraction                 |
| Pandas            | Tabular data representation         |

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
