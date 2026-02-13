import tkinter as tk
from tkinter import scrolledtext
import asyncio
import threading
import queue
import sys
import io
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# MCP Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Fix Encoding for the GUI terminal output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class MCPGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Legit MCP Research Assistant (Gemini + txtai)")
        self.root.geometry("800x650")

        # --- UI SETUP ---
        self.chat_display = scrolledtext.ScrolledText(root, state='disabled', font=("Segoe UI", 10), wrap=tk.WORD)
        self.chat_display.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)
        
        # Define Tags for colors
        self.chat_display.tag_config("user", foreground="#1a73e8", font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_config("gemini", foreground="#188038", font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_config("system", foreground="#70757a", font=("Segoe UI", 9, "italic"))

        input_frame = tk.Frame(root)
        input_frame.pack(padx=15, pady=(0, 15), fill=tk.X)

        self.user_input = tk.Entry(input_frame, font=("Segoe UI", 11))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", self.send_message)

        self.send_btn = tk.Button(input_frame, text="Ask AI", command=self.send_message, width=10)
        self.send_btn.pack(side=tk.RIGHT)

        self.status_label = tk.Label(root, text="Initializing MCP...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # --- BACKGROUND PROCESSING ---
        self.msg_queue = queue.Queue()
        self.resp_queue = queue.Queue()
        self.running = True

        self.thread = threading.Thread(target=self.start_async_loop, daemon=True)
        self.thread.start()

        self.root.after(100, self.check_responses)

    def append_chat(self, sender, message):
        self.chat_display.config(state='normal')
        tag = sender.lower()
        self.chat_display.insert(tk.END, f"{sender}: ", tag)
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')

    def send_message(self, event=None):
        text = self.user_input.get().strip()
        if text:
            self.append_chat("You", text)
            self.user_input.delete(0, tk.END)
            self.msg_queue.put(text)
            self.status_label.config(text="Thinking...")

    def check_responses(self):
        try:
            while True:
                sender, msg = self.resp_queue.get_nowait()
                if sender == "STATUS":
                    self.status_label.config(text=msg)
                else:
                    self.append_chat(sender, msg)
                    self.status_label.config(text="Ready")
        except queue.Empty:
            pass
        self.root.after(100, self.check_responses)

    def start_async_loop(self):
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(self.run_mcp_client())

    async def run_mcp_client(self):
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["server.py"],
            env=os.environ.copy()
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                tools_list = await session.list_tools()
                self.resp_queue.put(("STATUS", f"Connected to Server. {len(tools_list.tools)} tools ready."))
                
                # Setup Gemini
                gemini_tools = [{"name": t.name, "description": t.description, "parameters": t.inputSchema} for t in tools_list.tools]
                client = genai.Client(api_key=GEMINI_API_KEY)
                chat = client.chats.create(
                    model="gemini-2.5-flash-lite",
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(function_declarations=gemini_tools)],
                        system_instruction="You are a helpful assistant with access to PDF notes and Web Search."
                    )
                )

                while self.running:
                    if self.msg_queue.empty():
                        await asyncio.sleep(0.1)
                        continue
                    
                    prompt = self.msg_queue.get()
                    try:
                        # 1. Send to Gemini (Sync call inside Async thread is fine here)
                        response = chat.send_message(prompt)

                        # 2. Parallel Tool Call Handling
                        while response.candidates and response.candidates[0].content.parts:
                            parts = response.candidates[0].content.parts
                            f_calls = [p.function_call for p in parts if p.function_call]
                            
                            if not f_calls: break
                            
                            self.resp_queue.put(("System", f" Using tools: {[f.name for f in f_calls]}"))
                            
                            # Parallel Execution
                            tasks = [session.call_tool(f.name, arguments=f.args) for f in f_calls]
                            results = await asyncio.gather(*tasks)
                            
                            tool_responses = []
                            for f, res in zip(f_calls, results):
                                tool_responses.append(
                                    types.Part.from_function_response(name=f.name, response={"result": res.content[0].text})
                                )
                            
                            # Send back to Gemini
                            response = chat.send_message(tool_responses)
                        
                        self.resp_queue.put(("Gemini", response.text))

                    except Exception as e:
                        self.resp_queue.put(("System", f"Error: {str(e)}"))

if __name__ == "__main__":
    root = tk.Tk()
    app = MCPGuiApp(root)
    root.mainloop()