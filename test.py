from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import tool
import requests
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from models import TaggingAppointment, TaggingAppointmentSearch

_ = load_dotenv()

model = ChatOpenAI(temperature=0)


@tool(args_schema=TaggingAppointment)
def get_appointment_function(start: str, end: str, name: str, email: str) -> dict:

    """tag the piece of text with particular info, and then make the appointment, the start and the end time always with 2024"""

    params = {
        "eventTypeId": 949511,
        "start": start,
        "end": end,
        "responses": {
            "name": name,
            "email": email,
            "guests": [],
            "location": {
                "value": "Link",
                "optionValue": ""
            }
        },
        "metadata": {},
        "timeZone": "America/Bogota",
        "language": "en",
    }

    cal_api_key = os.getenv('CAL_API_KEY')

    url = f"https://api.cal.com/v1/bookings?apiKey={cal_api_key}"

    response_appointment = requests.post(url, json=params)

    print("Ok", response_appointment.json())
    print(f"Error {response_appointment.status_code}: {response_appointment.text}")

    if response_appointment.status_code == 200:
        print("Your appointment was successfully tagged")
    else:
        print("Your appointment was not successfully tagged")

    error_number = response_appointment.status_code
    print(error_number)

    url = f"https://api.cal.com/v1/event-types?apiKey={cal_api_key}"
    response_appointment = requests.get(url)
    print(response_appointment.json())

    return error_number


@tool()
def search_appointment():
    """when the user ask information for a day in order to schedule an appointment, the information of the start time is always in this format: YYYY-MM-DDTHH:MM:00.000Z, use this format and extract the day and month from the user input, the year is always 2024 and output the information from only that day"""

    cal_api_key = os.getenv('CAL_API_KEY')

    url = f"https://api.cal.com/v1/bookings?apiKey={cal_api_key}"

    response = requests.get(url)
    print(f"response booking {response.json()}")
    print(f"response status code {response.status_code}")

    data = response.json()

    start_times = [booking['startTime'] for booking in data['bookings']]

    print(start_times)

    return "ok"

@tool(args_schema=TaggingAppointmentSearch)
def get_appointment_info(id_appointment: str) -> str:
    """tag the piece of text with particular info, and search for the appointment with the given id"""

    api_key = os.getenv('CAL_API_KEY')

    booking_id = id_appointment
    url = f'https://api.cal.com/v1/bookings/{booking_id}'

    response = requests.get(url, params={'apiKey': api_key})

    print("Response:", response.json())
    print(f"Error {response.status_code}: {response.text}")

    return "ok appointment"


tool = [
    format_tool_to_openai_function(f) for f in [
        get_appointment_function, get_appointment_info, search_appointment
    ]
]

for tool in tool:
    print(f"this is tool {tool['name']}")

memory = SqliteSaver.from_conn_string(":memory:")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["action"]
        )
        self.tools = {t['name']: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        messages = self.model.invoke(messages)
        return {"messages": [messages]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


def active_agent(tool, memory, user_input):
    prompt = """You are helpful assistant, that helps the user to make an appointment or ask about one \
    tag the piece of text with particular info, the year of the request always is going to be 2024 \
    and if not explicitly provided do not guess ask again . Extract partial info \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that! \
    in order to have all the information you need to have the following information Name, Email, Start time. \
    if you are missing some of this information before you access to the tool ask for the missing info.\
    in the response return me the complete information (name, email, start time), confirming the information that would be use in the appointment.
    """

    abot = Agent(model, [tool], system=prompt, checkpointer=memory)

    messages = [HumanMessage(content=user_input)]
    thread = {"configurable": {"thread_id": "2"}}
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v)
            print(v["messages"][0].content)

            final_message = v["messages"][0].content

    return final_message


def model_function():
    functions = [
        format_tool_to_openai_function(f) for f in [
            get_appointment_function, get_appointment_info, search_appointment
        ]
    ]
    model_functions = ChatOpenAI(temperature=0).bind(functions=functions)

    prompt_agents_functions_holder = ChatPromptTemplate.from_messages([
        ("system", "You are helpful assistant, that helps the user to make an appointment or ask about, tag the piece of text with particular info, the year of the request always is going to be 2024, and if not explicitly provided do not guess. Extract partial info."),
        ("user", "{user_input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    chain_agents_holder = prompt_agents_functions_holder | model_functions | OpenAIFunctionsAgentOutputParser()

    agent_chain_run = RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
    ) | chain_agents_holder

    return agent_chain_run


def run_agent(final_message):
    intermediate_steps = []
    while True:
        result = agent_chain_run.invoke({
            "user_input": final_message,
            "intermediate_steps": intermediate_steps
        })
        if isinstance(result, AgentFinish):
            return result
        tool = {
            "get_appointment_function": get_appointment_function,
            "get_appointment_info": get_appointment_info,
            "search_appointment": search_appointment
        }[result.tool]
        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))


if __name__ == '__main__':
    user_input = "i want to make an appointment what time do you have, can you give options"
    agent_chain_run = model_function()
    run_agent(active_agent(tool, memory, user_input))
    # search_appointment()


# e.g i want to make an appointment my name is Andres Gonzalez, my email is leoracer@gmail.com and i want my appointment in august 30 from 2024 at 11:00 AM and the event type is 949511
