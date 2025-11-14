import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

# robust imports for langchain (support multiple versions)
PromptTemplate = None
LLMChain = None
SequentialChain = None

try:
    # common layout used in many examples
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        from langchain.prompts.prompt import PromptTemplate
    except Exception:
        # fall back to top-level (some langchain distributions expose symbols directly)
        try:
            from langchain import PromptTemplate  # noqa
        except Exception as e:
            raise ImportError(
                "PromptTemplate not found in the installed langchain package. "
                "Install a compatible langchain version (see README). Original error: "
                f"{e}"
            )

# LLMChain and SequentialChain: try a couple of plausible locations
try:
    from langchain.chains import LLMChain, SequentialChain
except Exception:
    try:
        # sometimes LLMChain lives in langchain.chains.llm
        from langchain.chains.llm import LLMChain
        from langchain.chains.sequential import SequentialChain
    except Exception:
        try:
            # last fallback - try top-level (rare)
            from langchain import LLMChain, SequentialChain  # noqa
        except Exception as e:
            raise ImportError(
                "LLMChain / SequentialChain not found in the installed langchain package. "
                "Please install a compatible 'langchain' package into the environment running Streamlit. "
                f"Original error: {e}"
            )

# huggingface and helper imports (unchanged)
try:
    from langchain_huggingface import HuggingFaceEndpoint
except Exception:
    raise ImportError("langchain_huggingface is not installed. pip install langchain-huggingface")

try:
    from huggingface_hub import login
except Exception:
    raise ImportError("huggingface_hub is not installed. pip install huggingface-hub")

# PDF reader will be handled by utils.py (see recommended change)
import PyPDF2
try:
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        from langchain.prompts.prompt import PromptTemplate
    except Exception:
        from langchain import PromptTemplate

load_dotenv()

key=os.getenv("HUGGING_FACE_KEY")
login(key)


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.5,# Pass temperature directly
    task="text-generation"
)


TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""


quiz_generation_prompt = PromptTemplate(
    input_variables=["text","number","subject","tone","response_json"],
    template=TEMPLATE
)

quiz_chain=LLMChain(llm=llm,prompt=quiz_generation_prompt,output_key="quiz",verbose=True)


TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis.
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt=PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2
    )


review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review",verbose=True)


generate_evaluate_chain=SequentialChain(
    chains=[quiz_chain,review_chain],
    input_variables=["text","number","subject","tone","response_json"],
    output_variables=["quiz","review"],
    verbose=True
    )
