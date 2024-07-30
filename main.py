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

import requests
import json


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

user_input = "i want to make an appointment my name is Andres Gonzalez, my email is andresfgonzalezr1996@gmail.com, my timezone is America/Chicago and mi bookingId is 1"


def get_appointment(user_input):
    prompt = f"""I need to extract the information about the user_input, also I have to fill out the next format and give please to me in a JSON format
    'bookingId' = Column(integer)
    'email' = Column(String)
    'name' = Column(String)
    'timeZone' = Column(String)
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

json_message = get_appointment(user_input)

cal_api_key = os.getenv('CAL_API_KEY')

url = f"https://api.cal.com/v1/attendees?apiKey={cal_api_key}"

headers = {"content-type": "application/json"}

data = json.loads(json_message)

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print("Ok", response.json())
else:
    print(f"Error {response.status_code}: {response.text}")
