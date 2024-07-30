#from langchain.agents import AgentExecutor, Agent
#from langchain_community.llms import OpenAI
#from langchain.prompts import PromptTemplate
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

#from langchain.prompts import ChatPromptTemplate
#from langchain.chat_models import ChatOpenAI
#from langchain.schema.output_parser import StrOutputParser


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

user_input = "i want to make an appointment with doctor andres july 29 at 4pm"


def get_appointment(user_input):
    prompt = f"""I need to extract the information about the user_input, also I have to fill out the next format and give please to me in a JSON format
    'doctor' = Column(String)
    'date' = Column(String)
    'time' = Column(String)
    the user_input is the following {user_input} from which you have to extract the information
    if the user_input has missing values return None for that values
    """

    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
            temperature=0,
      messages=[
        {"role": "user", "content": prompt}
      ]
    )

    mesage_response = response.choices[0].message.content

    return mesage_response

print(get_appointment(user_input))