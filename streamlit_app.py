import streamlit as st

import requests, os
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import json
from langchain.adapters.openai import convert_openai_messages
from requests.auth import HTTPBasicAuth
import subprocess
from langchain.output_parsers import PydanticOutputParser, StructuredOutputParser

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver

st.set_page_config(
    page_title="Chat App",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

class JMXScript(BaseModel):
    jmx_xml: str = Field(..., description="JMX test XML for the automated test")


@tool(args_schema=JMXScript)
def run_jmx_test(jmx_xml):
    """
    runs the JMeter tests and returns the results. Gives back and effort in case code excution fails
    """
    jmeter_path = "/opt/homebrew/Cellar/jmeter/5.6.3/bin/jmeter"
    jmx_file = "test_jmx.xml"

    with open(jmx_file, "w") as f:
        f.write(jmx_xml)

    result = subprocess.run([jmeter_path, "-n", "-t", jmx_file, "-l", "result.jtl"], capture_output=True)

    # Print the output of the JMeter run
    print(result.stdout.decode())
    print(result.stderr.decode())

    return result  # , result.stdout.decode(), result.stderr.decode()

tavily_tool = TavilySearchResults(max_results=4)
tools = [run_jmx_test, tavily_tool]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system=""):
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
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:  # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

prompt = """
You are a Senior Engineer/Developer specializing in performance testing using APICA and JMeter.
You are tasked with converting existing APICA test scripts provided as HAR file into a JMX file to run using JMeter. 

Step 1: Understand the APICA HAR file and list out the kep steps in the test
Step 2: Convert the key steps you identified into a JMX test XML
Step 3: Save the JMX test XML in memory
Step 4: Load the JMX text XML  from memory, run the test and get the results
    - In case of error, return back to step 2 to modify the errors and run again. Use the 'run_jmx_test' tool to run the test
    - At the end of this step you should have an executed JMX text
Step 5: If the run is successfull, return the below details
    - the JMX file content from memory
    - results of the JMX test run
    
You should only output the JMX XML and the results of the JMX test run. Nothing else.
"""

if "memory" not in st.session_state:
    st.session_state.memory = SqliteSaver.from_conn_string(":memory:")
    st.session_state.model = ChatOpenAI(model="gpt-4o")
    st.session_state.abot = Agent(st.session_state.model, tools, system=prompt)
    st.session_state.thread = {"configurable": {"thread_id": "1"}}

col1, col2 = st.columns(2)

with col1:
    apica = st.text_area("Apica JSON", key="apica", height=600)
    messages = [HumanMessage(content=apica)]

with col2:
    response = st.session_state.abot.graph.invoke({"messages": messages}, st.session_state.thread)
    st.text_area("JMX XML", key="jmx", value=response['messages'][-1].content, height=600)