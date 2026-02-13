import asyncio
import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# MCP Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

async def run_chat():
    if not GEMINI_API_KEY:
        print("Error: GOOGLE_API_KEY not found in .env")
        return

    # 1. Server Configuration
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=os.environ.copy()
    )

    print("🔌 Connecting to MCP Server...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 2. Discover Tools
            tools = await session.list_tools()
            print(f" Connected! Tools found: {[t.name for t in tools.tools]}")
            
            gemini_tools = []
            for tool in tools.tools:
                gemini_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                })

            # 3. Configure Gemini (Standard Sync Client)
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            # We use gemini-1.5-flash for speed and higher rate limits
            chat = client.chats.create(
                model="gemini-2.5-flash-lite",
                config=types.GenerateContentConfig(
                    tools=[types.Tool(function_declarations=gemini_tools)],
                    system_instruction="You are a helpful assistant. You can call multiple tools at once if needed."
                )
            )

            print("\nChat Started! (Type 'quit' to exit)\n")

            while True:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    break

                try:
                    # A. Send to Gemini (SYNC - Stable)
                    response = chat.send_message(user_input)
                    
                    # B. Handle Function Calls (PARALLEL ASYNC)
                    # We check if Gemini wants to run tools
                    if response.candidates and response.candidates[0].content.parts:
                        # Extract all function calls from the response
                        parts = response.candidates[0].content.parts
                        function_calls = [p.function_call for p in parts if p.function_call]
                        
                        if function_calls:
                            print(f"  Parallel Execution: Running {len(function_calls)} tools at once...")
                            
                            # 1. Create a list of async tasks
                            tasks = []
                            for fn in function_calls:
                                print(f"      -> Starting: {fn.name}")
                                # We schedule the tool call but don't wait yet
                                tasks.append(session.call_tool(fn.name, arguments=fn.args))
                            
                            # 2. Run them all together (Parallel!)
                            # This waits for the slowest one, not the sum of all
                            results = await asyncio.gather(*tasks)
                            
                            # 3. Prepare results for Gemini
                            response_parts = []
                            for fn, result in zip(function_calls, results):
                                print(f"      <- Finished: {fn.name}")
                                response_parts.append(
                                    types.Part.from_function_response(
                                        name=fn.name,
                                        response={"result": result.content[0].text}
                                    )
                                )
                            
                            # 4. Send results back to Gemini (SYNC)
                            response = chat.send_message(response_parts)

                    # C. Print Final Answer
                    print(f"Gemini: {response.text}\n")

                except Exception as e:
                    print(f" Error: {e}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_chat())