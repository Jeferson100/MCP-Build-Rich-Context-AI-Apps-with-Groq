from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
from groq import Groq
import os
import json



#MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
#MODEL = "deepseek-r1-distill-llama-70b"
MODEL = "qwen-qwq-32b"

client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY"),
)

nest_asyncio.apply()

load_dotenv()

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None
        #self.anthropic = Anthropic()
        self.available_tools: List[dict] = []
        

    async def process_query(self, query):
        messages = [{'role': 'user', 'content': query}]
        
        tools = [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]
            }
        } for tool in self.available_tools]
        
        response = client.chat.completions.create(
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
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_id = tool_call.id
                    
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    
                    # Executa a ferramenta
                    #result = execute_tool(tool_name, tool_args)
                    
                    result = await self.session.call_tool(tool_name, arguments=tool_args)
                    
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
    
    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="uv",  # Executable
            args=["run", "research_server.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()
    
                # List available tools
                response = await session.list_tools()
                
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])
                
                self.available_tools = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in response.tools]
    
                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()
  

if __name__ == "__main__":
    asyncio.run(main())