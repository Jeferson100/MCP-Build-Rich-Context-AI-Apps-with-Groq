
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import json
import asyncio
from groq import Groq
import os

#MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
MODEL = "deepseek-r1-distill-llama-70b"
#MODEL = "qwen-qwq-32b"


client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY"),
)

load_dotenv()

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = [] # new
        self.exit_stack = AsyncExitStack() # new
        #self.anthropic = Anthropic()
        self.groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.available_tools: List[ToolDefinition] = [] # new
        self.tool_to_session: Dict[str, ClientSession] = {} # new


    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            ) # new
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            ) # new
            await session.initialize()
            self.sessions.append(session)
            
            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])
            
            for tool in tools: # new
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self): # new
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise
    
    async def process_query(self, query):
        
        messages = [{'role':'user', 'content':query}]  
                                        
        tools = [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]
            }
        } for tool in self.available_tools]
        
        response = self.groq.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools = tools, # tools exposed to the LLM,
            max_tokens=2024   
        )
        process_query = True
        while process_query:
            assistant_message = response.choices[0].message
            
            # Verifica se há tool calls na resposta
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    if isinstance(tool_call.function.arguments, str):
                        tool_args = json.loads(tool_call.function.arguments)
                    else:
                        tool_args = tool_call.function.arguments
                    
                    tool_id = tool_call.id
                    
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    
                    # Verifica se a ferramenta existe
                    if tool_name not in self.tool_to_session:
                        print(f"Ferramenta {tool_name} não encontrada")
                        continue
                
                    
                    session = self.tool_to_session[tool_name] # new
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    
                    result_content = str(result.content) if hasattr(result, 'content') else str(result)

        
                    # Adiciona a mensagem do assistente e o resultado da ferramenta
                    messages.append({
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [tool_call]
                    })
                    
                    messages.append({
                        'role': 'tool',
                        'content': result_content,
                        'tool_call_id': tool_id
                    })
                    
                    # Faz nova chamada para processar o resultado
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=tools,
                        max_tokens=2024
                    )
            else:
                # Se não houver tool calls, imprime a resposta e encerra
                print(assistant_message.content)
                process_query = False
                
            messages.append({
                'role': 'assistant',
                'content': assistant_message.content
            })

    
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self): # new
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        # the mcp clients and sessions are not initialized using "with"
        # like in the previous lesson
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers() # new! 
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup() #new! 


if __name__ == "__main__":
    asyncio.run(main())
