#all the import you need 
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool,wiki_tool, save_to_txt,save_tool
import os

#to use th API key in the .env file 
load_dotenv()

#how you want your model to be 
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    
#call the API you want to use 
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=anthropic_api_key
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
     
      (
         
     "system",
     """
     you are a reseach assistant that will help generate a research paper.
     Answer the user query and use neccessary tools.
     Wrap the output in this format and provide no other text\n{format_instruction}
     """,   
      ) , 
      ("placeholder", " {chat_history}"),
      ("human", "{query}"),
      ("placeholder", " {agent_scratchpad}"),
    ]
 #  format instruction promtvariable   
).partial(format_instructions=parser.get_format_instructions())
 
# at this level we have the promt the perser and the llm now lets create the agent 
tools = [ search_tool,wiki_tool,save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor( agent=agent,tools=tools, verbose=True)
query = input("what can i help you research")
raw_response = agent_executor.invoke({"query": "what is the capital of france?"})


try:   
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print=(structured_response)
except Exception as e:
    print("Error parsing response", e, "raw_response -", raw_response)

 
















