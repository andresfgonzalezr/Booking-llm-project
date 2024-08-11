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


model = ChatOpenAI(temperature=0)


class TaggingAppointment(BaseModel):
    """tag the piece of text with particular info."""
    eventTypeId: int = Field(description="this is the event type")
    start: str = Field(description="the start time, this has to be in the format 'YYYY-MM-DDTHH:MM:00.000Z'")
    end: str = Field(description="the end time, has to be in the format 'YYYY-MM-DDTHH:MM:00.000Z', also has to be 30 minutes difference between the start and end dates.")
    name: str = Field(description="the name of the person that is making the appointment")
    email: str = Field(description="the email of the person that is making the appointment")
    time_zone: str = Field(description="the time zone of the person that is making the appointment e.g America/Bogota")
    value: Optional[str] = Field(description="the way the appointment is going to be done e.g cal video, zoom, presentail")


convert_pydantic_to_openai_function(TaggingAppointment)

tagging_functions = [convert_pydantic_to_openai_function(TaggingAppointment)]

prompt_tagging_appointment = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed, if not explicitly provided do not guess, extract partial info"),
    ("user", "{user_input}")
])

model_with_tagging_function = model.bind(
    functions=tagging_functions,
    function_call={"name": "TaggingAppointment"}
)

tagging_chain = prompt_tagging_appointment | model_with_tagging_function # | JsonOutputFunctionsParser()

print(tagging_chain.invoke({"user_input": "i want to make an appointment my name is Andres Gonzalez, my email is leoracer@gmail.com, my timezone is America/Bogota and i want my appointment in august 9 from 2024 at 13:00 AM and the event type is 949511"}))

json_message = tagging_chain.invoke({"user_input": "i want to make an appointment my name is Andres Gonzalez, my email is leoracer@gmail.com, my timezone is America/Bogota and i want my appointment in august 9 from 2024 at 13:00 AM and the event type is 949511"})

data = json_message.additional_kwargs
arguments_data = data["function_call"]["arguments"]

json_data = json.loads(arguments_data)


@tool
def get_appointment_function(appointment: TaggingAppointment) -> dict:
    """tag the piece of text with particular info."""
    eventTypeId: int = Field(description="this is the event type")
    start: str = Field(description="the start time, this has to be in the format 'YYYY-MM-DDTHH:MM:00.000Z'")
    end: str = Field(description="the end time, has to be in the format 'YYYY-MM-DDTHH:MM:00.000Z', also has to be 30 minutes difference between the start and end dates.")
    name: str = Field(description="the name of the person that is making the appointment")
    email: str = Field(description="the email of the person that is making the appointment")
    time_zone: str = Field(description="the time zone of the person that is making the appointment e.g America/Bogota")
    value: Optional[str] = Field(description="the way the appointment is going to be done e.g cal video, zoom, presentail")



    cal_api_key = os.getenv('CAL_API_KEY')

    url = f"https://api.cal.com/v1/bookings?apiKey={cal_api_key}"


    response_appointment = requests.post(url, json=appointment.dict)

    print("Ok", response_appointment.json())
    print(f"Error {response_appointment.status_code}: {response_appointment.text}")

    url = f"https://api.cal.com/v1/event-types?apiKey={cal_api_key}"
    response_appointment = requests.get(url)
    print(response_appointment.json())

format_tool_to_openai_function(get_appointment_function)

print(get_appointment_function({"user_input": "i want to make an appointment my name is Andres Gonzalez, my email is leoracer@gmail.com, my timezone is America/Bogota and i want my appointment in august 9 from 2024 at 13:00 AM and the event type is 949511"}))


class TaggingAppointmentSearch(BaseModel):
    """tag the piece of text with particular info."""
    id: int = Field(description="this is the id")


@tool
def get_appointment_info(user_input: str) -> dict:
    """tag the piece of text with particular info."""
    id: int = Field(description="this is the id")

    api_key = os.getenv('API_KEY')

    booking_id = '949511'
    url = f'http://localhost:3002/v1/bookings/{booking_id}'

    response = requests.get(url, params={'apiKey': api_key})

    print("Response:", response.json())
    print(f"Error {response.status_code}: {response.text}")


functions = [
    format_tool_to_openai_function(f) for f in [
        get_appointment_function, get_appointment_info
    ]
]
model_functions = ChatOpenAI(temperature=0).bind(functions=functions)

prompt_agent_functions = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant, that helps the user to make an appointment or ask about one"),
    ("user", "{input}"),
])
chain_agents = prompt_agent_functions | model_functions | OpenAIFunctionsAgentOutputParser()

result = chain_agents.invoke({"user_input": "hi"})


def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "get_appointment_function": get_appointment_function,
            "get_appointment_info": get_appointment_info,
        }
        return tools[result.tool].run(result.tool_input)


chain_agents = prompt_agent_functions | model_functions | OpenAIFunctionsAgentOutputParser() | route


