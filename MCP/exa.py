import asyncio
import warnings
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

warnings.filterwarnings("ignore")
load_dotenv()


class MCPAgent:
    def __init__(self):
        self.client = None
        self.agent = None

    async def init(self):
        # IMPORTANT: initialize only once
        if self.agent is not None:
            return

        if self.client is None:
            self.client = MultiServerMCPClient(
                {
                    "exa": {
                        "transport": "streamable_http",
                        "url": "https://mcp.exa.ai/mcp",
                    }
                }
            )
        SYSTEM_PROMPT = """
            You are a technical documentation assistant.

            You must follow these rules:

            1. Use Exa ONLY when the question is about:
            - official API documentation
            - frameworks (FastAPI, LangChain, Django, etc.)
            - programming references
            - GitHub repositories or code-level explanations

            2. Never use Exa for:
            - news
            - general web search
            - opinions
            - recipes
            - entertainment content

            3. If unsure, answer from your internal knowledge first.

            4. Prefer official documentation sources over blogs or articles.

            Keep answers short, precise, and developer-focused.
            """
        tools = await self.client.get_tools()

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        self.agent = create_agent(model=llm, tools=tools,system_prompt=SYSTEM_PROMPT,verbose=True)

    async def run(self, query: str):
        await self.init()

        result = await self.agent.ainvoke(
            {"messages": [("user", query)]}
        )

        return result["messages"][-1].text


agent_instance = MCPAgent()

def run_sync(query: str):
    return asyncio.run(agent_instance.run(query))