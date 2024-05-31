# ----------------------------------- Import libs ----------------------------------- #

import os, openai
from configparser import ConfigParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.utilities.dataforseo_api_search import DataForSeoAPIWrapper

# ----------------------------------- Load from config ----------------------------------- #

config = ConfigParser()

try:
  config.read('config.ini')
except:
  print('config.ini format error')
  raise SystemExit()

openai.api_type = config['OpenAI']['api_type']
openai.api_key = config['OpenAI']['api_key']
openai.api_base = config['OpenAI']['api_base']
openai.api_version = config['OpenAI']['api_version']

os.environ["DATAFORSEO_LOGIN"] = config['DataForSeoAPIWrapper']['DATAFORSEO_LOGIN']
os.environ["DATAFORSEO_PASSWORD"] = config['DataForSeoAPIWrapper']['DATAFORSEO_PASSWORD']

# ----------------------------------- Model defination ----------------------------------- #

model = AzureChatOpenAI(
        openai_api_version=openai.api_version,
        openai_api_base=openai.api_base,
        openai_api_key=openai.api_key,
        azure_deployment=config['OpenAI']['model_name']
)

# ----------------------------------- Prompt template ----------------------------------- #

prompt = ChatPromptTemplate.from_template(
    """You are a friendly chatbot specialized in providing weather reports. When users ask you a question, they will also specify the language in which you should respond and list traits that you should embody in your response.

Input:
- Traits: {traits}
- Language: {language}
- Question from User: {question}
- Context: {context}
- AI Response:

Instructions:
1) Use the provided context to directly answer weather-related questions. Avoid introducing unnecessary disclaimers about your capabilities when the question is clearly about weather.
2) If a question is clearly not about the weather, mention That you can only provide weather-related information using the traits available. YOU MUST NOT GIVE ANY OTHER INFORMATION.
3) Always check the relevance of the question to weather information before deciding on your response approach."""
)

# ----------------------------------- Langchain language expression ----------------------------------- #

chain = prompt | model | StrOutputParser()

# ----------------------------------- Generate response function ----------------------------------- #

def generate_response(language: str, traits: list, user_prompt: str) -> str:
    """
    Generates a response based on the specified user prompt and context.

    Parameters:
        language (str): The language in which the response should be generated.
        traits (list): A list of traits that the response should embody.
        user_prompt (str): The user's prompt to which the bot will respond.

    Returns:
        str: A string representing the AI-generated response based on the given context.
    """
    try:
        json_wrapper = DataForSeoAPIWrapper(
            top_count=3,
            json_result_types=["organic", "local_pack"],
            json_result_fields=["title", "description", "type", "text"],
        )

        modified_user_prompt = user_prompt + "[NOTE:-strictly follow the specified Instructions while replying]"
        context = json_wrapper.results(modified_user_prompt)
        
        response = chain.invoke(
            {"question": modified_user_prompt, "context": context, "language": language, "traits": traits})

        return response, context
    except Exception as e:
        print(e)

        return "", {}