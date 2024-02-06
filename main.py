from fastapi import FastAPI, Depends
from pydantic import BaseModel
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, PromptHelper, StorageContext, load_index_from_storage
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY')
cohere_api_key = os.environ.get('COHERE_API_KEY')
app = FastAPI()

class Question(BaseModel):
    input_text: str

def init_index(directory_path):
    os.environ["OPENAI_API_KEY"] = openai_api_key

    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_overlap_ratio = 0.5  # Corrected parameter

    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio, chunk_size_limit=max_chunk_overlap)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    index.storage_context.persist("index_directory")

    return index

def chatbot(input_text):
    # Rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="index_directory")

    # Load index from the storage context
    loaded_index = load_index_from_storage(storage_context)

    new_query_engine = loaded_index.as_query_engine()
    response = new_query_engine.query(input_text)
    return response.response

openai.api_key = openai_api_key

def generate_chatgpt_response(user_prompt):
    # User message only, without a system message
    user_message = {"role": "user", "content": user_prompt}

    # Create a list of messages for the Chat API
    messages = [user_message]

    try:
        # Make a request to the Chat API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Extract the assistant's reply from the response
        assistant_reply = response.choices[0].message.content
        return assistant_reply

    except Exception as e:
        # Handle exceptions and return an error message
        return f"Error generating chat response: {str(e)}"
# Function to execute the chatbot and ChatGPT functions in parallel
def execute_in_parallel(user_prompt):
    with ThreadPoolExecutor() as executor:
        future_chatbot = executor.submit(chatbot, user_prompt)
        future_chatgpt = executor.submit(generate_chatgpt_response, user_prompt)

    response_chatbot = future_chatbot.result()
    response_chatgpt = future_chatgpt.result()

    return {"response_chatbot": response_chatbot, "response_chatgpt": response_chatgpt}

@app.post("/ask/")
def ask_question(question: Question):
    # Execute chatbot and ChatGPT functions in parallel
    parallel_responses = execute_in_parallel(question.input_text)

    return {"response_chatbot": parallel_responses["response_chatbot"], "response_chatgpt": parallel_responses["response_chatgpt"]}

# Initialize index
init_index("docs")