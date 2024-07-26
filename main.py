from langchain.agents import AgentExecutor, Agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


details_prompt = PromptTemplate("User wants to book an appointment. Ask for details, Name, Phone number, Email, ")
availability_prompt = PromptTemplate("Check if the requested time is available.")
confirmation_prompt = PromptTemplate("Confirm the appointment with the user.")


details_agent = Agent(OpenAI(), details_prompt)
availability_agent = Agent(OpenAI(), availability_prompt)
confirmation_agent = Agent(OpenAI(), confirmation_prompt)


class AppointmentAgentExecutor(AgentExecutor):
    def __init__(self):
        self.agents = [details_agent, availability_agent, confirmation_agent]
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
