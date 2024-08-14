import os
import openai
import json
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.agents import tool
import requests
from langchain_core.messages import AIMessage
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish
# from langchain.tools import tool
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from models import TaggingAppointment, TaggingAppointmentSearch


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

model = ChatOpenAI(temperature=0)


convert_pydantic_to_openai_function(TaggingAppointment)

tagging_functions = [convert_pydantic_to_openai_function(TaggingAppointment)]


@tool(args_schema=TaggingAppointment)
def get_appointment_function(eventTypeId: int, start: str, end: str, name: str, email: str, time_zone: str) -> dict:

    """tag the piece of text with particular info, and then make the appointment"""

    params = {
        "eventTypeId": eventTypeId,
        "start": start,
        "end": end,
        "name": name,
        "email": email,
        "time_zone": time_zone
    }

    cal_api_key = os.getenv('CAL_API_KEY')

    url = f"https://api.cal.com/v1/bookings?apiKey={cal_api_key}"

    response_appointment = requests.post(url, params=params)

    print("Ok", response_appointment.json())
    print(f"Error {response_appointment.status_code}: {response_appointment.text}")

    url = f"https://api.cal.com/v1/event-types?apiKey={cal_api_key}"
    response_appointment = requests.get(url)
    print(response_appointment.json())

    return response_appointment.json()


format_tool_to_openai_function(get_appointment_function)

# print(get_appointment_function({"user_input": "i want to make an appointment my name is Andres Gonzalez, my email is leoracer@gmail.com, my timezone is America/Bogota and i want my appointment in august 9 from 2024 at 13:00 AM and the event type is 949511"}))


@tool(args_schema=TaggingAppointmentSearch)
def get_appointment_info(id_appointment: str) -> str:
    """tag the piece of text with particular info, and search for the appointment with the given id"""

    api_key = os.getenv('API_KEY')

    booking_id = id_appointment
    url = f'https://api.cal.com/v1/bookings/{booking_id}'

    response = requests.get(url, params={'apiKey': api_key})

    print("Response:", response.json())
    print(f"Error {response.status_code}: {response.text}")

    return "ok appointment"


format_tool_to_openai_function(get_appointment_info)


functions = [
    format_tool_to_openai_function(f) for f in [
        get_appointment_function, get_appointment_info
    ]
]
model_functions = ChatOpenAI(temperature=0).bind(functions=functions)

prompt_agent_functions = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant, that helps the user to make an appointment or ask about one, tag the piece of text with particular info"),
    ("user", "{user_input}"),
])

chain_agents = prompt_agent_functions | model_functions | OpenAIFunctionsAgentOutputParser()

result = chain_agents.invoke({"user_input": "i want to make an appointment my name is Andres Gonzalez, my email is leoracer@gmail.com, my timezone is America/Bogota and i want my appointment in august 9 from 2024 at 13:00 AM and the event type is 949511"})


prompt_agents_functions_holder = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant, that helps the user to make an appointment or ask about one, tag the piece of text with particular info"),
    ("user", "{user_input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

chain_agents_holder = prompt_agents_functions_holder | model_functions | OpenAIFunctionsAgentOutputParser()

result1 = chain_agents_holder.invoke({
    "user_input": "i want to make an appointment my name is Andres Gonzalez, my email is leoracer@gmail.com, my timezone is America/Bogota and i want my appointment in august 9 from 2024 at 13:00 AM and the event type is 949511",
    "agent_scratchpad": []
})


observation = get_appointment_function(result1.tool_input)


format_to_openai_functions([(result1, observation)])

result2 = chain_agents_holder.invoke({
    "user_input": "",
    "agent_scratchpad": format_to_openai_functions([(result1, observation)])
})

def run_agent(user_input):
    intermediate_steps = []
    while True:
        result = chain_agents_holder.invoke({
            "user_input": user_input,
            "agent_scratchpad": format_to_openai_functions(intermediate_steps)
        })
        if isinstance(result, AgentFinish):
            return result
        tool = {
            "get_appointment_function": get_appointment_function,
            "get_appointment_info": get_appointment_info
        }[result.tool]
        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))


agent_chain_run = RunnablePassthrough.assign(
    agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
) | chain_agents_holder


def run_agent(user_input):
    intermediate_steps = []
    while True:
        result = agent_chain_run.invoke({
            "user_input": user_input,
            "intermediate_steps": intermediate_steps
        })
        if isinstance(result, AgentFinish):
            return result
        tool = {
            "get_appointment_function": get_appointment_function,
            "get_appointment_info": get_appointment_info,
        }[result.tool]
        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))

