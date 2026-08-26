import warnings
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

warnings.filterwarnings("ignore")

async def get_agent():
    load_dotenv()

    client = MultiServerMCPClient(
            {
                "exa": {
                    "transport": "streamable_http",
                    "url": "https://mcp.exa.ai/mcp",
                }
            }
        )

    tools = await client.get_tools()

    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash"
        )

    agent = create_agent(
            model=llm,
            tools=tools,
        )

    return agent
    # result = await agent.ainvoke(
    #     {
    #         "messages": [
    #             (
    #                 "user",
    #                 "How do I use FastAPI lifespan events?"
    #             )
    #         ]
    #     }
    # )
    # print(result["messages"][-1].text)
    

# if __name__ == "__main__":
#     asyncio.run(main())
