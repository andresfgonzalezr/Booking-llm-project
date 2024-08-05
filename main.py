#from langchain.agents import AgentExecutor, Agent
#from langchain_community.llms import OpenAI
#from langchain.prompts import PromptTemplate
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain_core.runnables import RunnableSequence
#from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
#from langchain.schema.output_parser import StrOutputParser

import requests
import json


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

user_input = "i want to make an appointment my name is Andres Gonzalez, my email is leoracer@gmail.com, my timezone is America/Bogota and i want my appointment in august 9 from 2024 at 13:00 AM and the event type is 949511"


def get_appointment(user_input):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    prompt_appointment = ChatPromptTemplate.from_template(f"""I need to extract the information from the given user_input and fill out the following format in JSON:
    {{
        "eventTypeId": 000000,
        "start": "2024-08-01T13:00:00.000Z",
        "end": "2024-08-01T13:00:00.000Z",
        "responses": {{
            "name": "",
            "email": "",
            "guests": [],
            "location": {{
                "value": "cal video",
                "optionValue": ""
            }}
        }},
        "metadata": {{}},
        "timeZone": "America/Chicago",
        "language": "en"
    }}
    The user_input is: {user_input}. 
    Please extract the necessary information to fill the above fields. For the 'start' and the 'end' field, use the format 'YYYY-MM-DDTHH:MM:00.000Z'. If the year is not mentioned, default to 2024, also the 'end' always 30 minutes diference between the start and end dates.
    """)

    chain_one = prompt_appointment | llm

    prompt_confirmation = ChatPromptTemplate.from_template("""Based on the extracted appointment details, create a confirmation message for the user.
    Here are the details:
    {appointment_details}
    Please generate a message confirming the appointment and providing any necessary information.
    """)

    chain_two = prompt_confirmation | llm

    result = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)

    return result


print(get_appointment(user_input))

json_message = get_appointment(user_input)

cal_api_key = os.getenv('CAL_API_KEY')

url = f"https://api.cal.com/v1/bookings?apiKey={cal_api_key}"

data = json.loads(json_message)

response = requests.post(url, json=data)

print("Ok", response.json())
print(f"Error {response.status_code}: {response.text}")

url = f"https://api.cal.com/v1/event-types?apiKey={cal_api_key}"
response = requests.get(url)
print(response.json())
