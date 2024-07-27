from langchain.agents import AgentExecutor, Agent
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
import openai
from dotenv import load_dotenv, find_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()



appointment_prompt = ChatPromptTemplate.from_template("Confirm if the user wants to book an appointment or not")
details_prompt = PromptTemplate("User wants to book an appointment. Ask for details, Name, Phone number, Email, ")
availability_prompt = PromptTemplate("Check if the requested time is available.")
confirmation_prompt = PromptTemplate("Confirm the appointment with the user.")


appointment_agent = OpenAI(OpenAI(), appointment_prompt)
details_agent = Agent(OpenAI(), details_prompt)
availability_agent = Agent(OpenAI(), availability_prompt)
confirmation_agent = Agent(OpenAI(), confirmation_prompt)


class AppointmentAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agents = [appointment_agent, details_agent, availability_agent, confirmation_agent]
        self.data = {}

    def run(self, input_text):
        for agent in self.agents:
            output = agent.run(input_text)
            print(output)
            input_text = output
        return output


if __name__ == "__main__":
    system = AppointmentAgentExecutor()
    user_input = "I would like to book a doctor's appointment."
    result = system.run(user_input)
    print("Final Output:", result)
