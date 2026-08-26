import asyncio
from typing import TypedDict, List,Annotated
# from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage,BaseMessage,AIMessage
from dotenv import load_dotenv
import os
from langsmith import traceable
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


load_dotenv()
# Read system prompt from file
def load_system_prompt():
    prompt_file = os.path.join(os.path.dirname(__file__), '..', 'systemPrompt.txt')
    with open(prompt_file, 'r') as f:
        return f.read().strip()

SYSTEM_PROMPT = load_system_prompt()
# -------------------------
# STATE
# -------------------------
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    route: str
    search_query: str
    tool_result: str
    final_answer: str
    summary: str


REWRITE_PROMPT = """
You are a query optimizer for technical documentation search.

TOOL USAGE RULES:
You MUST use Exa MCP ONLY for:
- official API documentation
- programming frameworks (FastAPI, LangChain, Django, etc.)
- GitHub repositories
- SDKs, libraries, and developer references

SOURCE POLICY:
- Prefer official documentation and GitHub sources
- Avoid blogs unless no official source exists

STYLE:
- Be concise
- Be technical
- Do not add unnecessary explanation

Convert the user question into a precise search query for official documentation.
Return ONLY the optimized query.
"""

ANSWER_PROMPT = """
You are a technical documentation assistant.

Use ONLY the provided tool results.
Do not hallucinate.
Be concise and accurate.
"""


# -------------------------
# MCP CLIENT
# -------------------------
async def get_tools():
    client = MultiServerMCPClient(
        {
            "exa": {
                "transport": "streamable_http",
                "url": "https://mcp.exa.ai/mcp",
            }
        }
    )
    return await client.get_tools()


# -------------------------
# BUILD GRAPH
# -------------------------
async def build_graph(checkpointer):

    # llm_router =ChatDeepSeek(model="deepseek-chat", temperature=0.7)
    # llm_rewriter = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
    # llm_answer = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
    llm_router = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    llm_rewriter = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    llm_answer = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

    # -------------------------
    # NODE 1: CLASSIFIER (ALLOW / BLOCK)
    # -------------------------
    @traceable(name="classifier")
    async def classifier_node(state: State):

        response = await llm_router.ainvoke([
            SystemMessage(content="""
                Classify the query.

                Return ONLY one word:
                ALLOW → software engineering / coding / documentation
                BLOCK → everything else (recipes, news, entertainment, general knowledge, etc.)
                """),
            HumanMessage(content=state["query"])
        ])

        label = response.content.strip().upper()

        return {"route": label}


    # -------------------------
    # NODE 2: REWRITE QUERY
    # -------------------------
    @traceable(name="rewrite")
    async def rewrite_node(state: State):

        response = await llm_rewriter.ainvoke([
            SystemMessage(content=REWRITE_PROMPT),
            HumanMessage(content=state["query"])
        ])

        return {"search_query": response.content}


    # -------------------------
    # NODE 3: TOOL EXECUTION (FORCED MCP)
    # -------------------------
    # @traceable(name="tool")
    # async def tool_node(state: State):
    #     tools = await get_tools()

    #     search_tool = tools[0]

    #     result = await search_tool.ainvoke({
    #         "query": state["search_query"]
    #     })

    #     return {
    #         "tool_result": result.content if hasattr(result, "content") else str(result)
    #     }
    @traceable(name="tool")
    async def tool_node(state: State):
        tools = await get_tools()
        search_tool = tools[0]

        result = await search_tool.ainvoke({
            "query": state["search_query"]
        })

        # Handle both string and list-of-blocks content
        if isinstance(result.content, list):
            tool_text = "\n".join(
                block["text"]
                for block in result.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        elif isinstance(result.content, str):
            tool_text = result.content
        else:
            tool_text = str(result.content)

        return {"tool_result": tool_text}

    # -------------------------
    # NODE 4: ANSWER GENERATION
    # -------------------------
    @traceable(name="answer")
    async def answer_node(state: State):

        messages = [SystemMessage(content=ANSWER_PROMPT)]

        # Add conversation summary if it exists
        if state.get("summary"):
            messages.append(
                SystemMessage(
                    content=f"""Conversation Summary:{state['summary']}"""))

        messages.extend(state["messages"][-10:])
        # Current query + tool results
        messages.append(
            HumanMessage(
                content=f"""
                Search Results:
                {state['tool_result']}"""))

        response = await llm_answer.ainvoke(messages)

        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content)]
        }

    # -------------------------
    # NODE 5: REFUSAL NODE
    # -------------------------
    async def refusal_node(state: State):
        return {
            "final_answer": "I can only assist with software development and technical documentation queries.",
            "messages": [
        AIMessage(content="I can only assist with software development and technical documentation queries.")]
        }
    async def summarize_node(state: State):

        messages = state["messages"]

        # don't summarize yet
        if len(messages) < 40:
            return {}
        conversation = "\n".join(f"{msg.type}: {msg.content}"for msg in messages[:-10])
        summary_prompt = f"""
        Existing Summary:
        {state.get('summary', '')}

        Update the summary using the conversation below.

        Preserve:
        - goals
        - decisions
        - preferences
        - important facts

        Conversation:
        {conversation}
        """

        response = await llm_answer.ainvoke(summary_prompt)

        return {
            "summary": response.content,
            "messages": messages[-10:]
        }

    # -------------------------
    # ROUTER
    # -------------------------
    def route_decision(state: State):
        return state["route"]


    # -------------------------
    # GRAPH
    # -------------------------
    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("refusal", refusal_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("classifier")

    graph.add_conditional_edges(
        "classifier",
        route_decision,
        {
            "ALLOW": "rewrite",
            "BLOCK": "refusal"
        }
    )

    graph.add_edge("rewrite", "tool")
    graph.add_edge("tool", "answer")
    graph.add_edge("answer","summarize")
    graph.add_edge("summarize",END)
    graph.add_edge("refusal", END)

    return graph.compile(checkpointer=checkpointer)

app = None
checkpointer = None
checkpointer_cm = None

async def get_checkpointer():
    global checkpointer, checkpointer_cm

    if checkpointer is not None:
        return checkpointer

    DB_URI = os.getenv("DATABASE_URL")
    if not DB_URI:
        raise ValueError("DATABASE_URL not found")

    checkpointer_cm = AsyncPostgresSaver.from_conn_string(DB_URI)

    checkpointer = await checkpointer_cm.__aenter__()

    await checkpointer.setup()

    return checkpointer

async def get_app():
    global app
    cp = await get_checkpointer()
    if app is not None:
        return app

    app = await build_graph(cp)

    return app
# async def get_app():
#     DB_URI = os.getenv("DATABASE_URL")

#     checkpointer_cm = AsyncPostgresSaver.from_conn_string(DB_URI)

#     checkpointer = await checkpointer_cm.__aenter__()

#     await checkpointer.setup()

#     return await build_graph(checkpointer)

# -------------------------
# RUN FUNCTION
# -------------------------
def run_sync(query: str,thread_id : str):

    async def _run():
        app = await get_app()
        config = {
            "configurable": {
                "thread_id": thread_id
            },
            "metadata": {
                "chat_id": thread_id

            }
        }
        result = await app.ainvoke({
            "query": query,
            "messages": [
            HumanMessage(content=query)
        ]
        },config=config
        )

        return result["final_answer"]

    return asyncio.run(_run())

