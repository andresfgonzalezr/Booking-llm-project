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



from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


def get_appointment(user_input):

    """Get the informatión from the user_input, give it to me in json format, For the 'start' and the 'end' field, use the format 'YYYY-MM-DDTHH:MM:00.000Z'. If the year is not mentioned, default to 2024, also the 'end' always 30 minutes difference between the start and end dates."""
    appointment_info = {
        "user_input": user_input,
        "eventTypeId": 000000,
        "start": "2024-08-01T13:00:00.000Z",
        "end": "2024-08-01T13:00:00.000Z",
        "responses": {
            "name": "",
            "email": "",
            "guests": [],
            "location": {
                "value": "cal video",
                "optionValue": ""
            }
        },
        "metadata": {},
        "timeZone": "America/Chicago",
        "language": "en"
    }

    return json.dumps(appointment_info)


functions = [
    {
        "name": "get_appointment",
        "description": "Get the information from the user_input, give it to me in JSON format. For the 'start' and 'end' field, use the format 'YYYY-MM-DDTHH:MM:00.000Z'. If the year is not mentioned, default to 2024, also the 'end' always 30 minutes difference between the start and end dates.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_input": {
                    "type": "string",
                    "description": "This is where all the information is, from where it has to be extracted for the JSON."
                }
            },
            "required": ["user_input"]
        }
    }
]

messages = [
    {
        "role": "user",
        "content": "i want to make an appointment my name is Andres Gonzalez, my email is leoracer@gmail.com, my timezone is America/Bogota and i want my appointment in august 9 from 2024 at 13:00 AM and the event type is 949511"
    }
]

response = openai.chat.completions.create(
    model="gpt-4",
    messages=messages,
    functions=functions,
    function_call="auto"
)

response_message = response.choices[0].message.function_call

messages.append(response.choices[0].message)

user_input = response_message

# print(get_appointment(user_input))


class GetAppointment(BaseModel):
    """I need to extract the information from the given user_input and fill out the following format in JSON:
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
    """
    user_input: str = Field(description="This is where all the information is, from where it has to be extracted for the JSON.")


model = ChatOpenAI(temperature=0)

appointment_function = convert_pydantic_to_openai_function(GetAppointment)

model_user_input = model.invoke("i want to make an appointment my name is Andres Gonzalez, my email is leoracer@gmail.com, my timezone is America/Bogota and i want my appointment in august 9 from 2024 at 13:00 AM and the event type is 949511", functions=[appointment_function])

# print(model_user_input)

model_with_function = model.bind(functions=[appointment_function])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps extrating the information from the user_input, in order to make an appointment with a doctor"),
    ("user", "{user_input}")
])

chain = prompt | model_with_function

response_chain = chain.invoke({"user_input": "i want to make an appointment my name is Andres Gonzalez, my email is leoracer@gmail.com, my timezone is America/Bogota and i want my appointment in august 9 from 2024 at 13:00 AM and the event type is 949511"})

# print(response_chain)


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
        get_appointment(), get_appointment_info
    ]
]
model = ChatOpenAI(temperature=0).bind(functions=functions)
