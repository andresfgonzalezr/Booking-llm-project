from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import AgentExecutor, Agent
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.tools import tool

@tool
def get_an_appointment(query: str) -> str:
    """Confirm if the user wants to book an appointment or not"""
    prompt = PromptTemplate(query=query)
    return prompt.ask()

@tool
def get_details(query: str) -> str:
    """User wants to book an appointment. Ask for details, Name, Phone number, Email, the date the person wants to make the appointment and the hour"""
    prompt = PromptTemplate(query=query)
    return prompt.ask()

@tool
def get_available(query: str) -> str:
    """Check if the requested time is available."""
    prompt = PromptTemplate(query=query)
    return prompt.ask()

@tool
def get_confirmation(query: str) -> str:
    """Confirm the appointment with the user."""
    prompt = PromptTemplate(query=query)
    return prompt.ask()