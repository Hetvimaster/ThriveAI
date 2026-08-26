import asyncio
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from dotenv import load_dotenv
import os
from langsmith import traceable
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

load_dotenv()

# -------------------------
# SYSTEM PROMPT
# -------------------------
def load_system_prompt():
    prompt_file = os.path.join(os.path.dirname(__file__), '..', 'systemPrompt.txt')
    with open(prompt_file, 'r') as f:
        return f.read().strip()

SYSTEM_PROMPT = load_system_prompt()

# -------------------------
# HELPER: safely extract text from any LLM/tool response
# Handles: plain string, list of content blocks, or anything else
# -------------------------
def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif "text" in block:
                    parts.append(block["text"])
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)

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
    llm_router = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    llm_rewriter = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    llm_answer = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

    # -------------------------
    # NODE 1: CLASSIFIER
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
        # FIX: use extract_text so .strip() never fails on a list
        label = extract_text(response.content).strip().upper()
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
        return {"search_query": extract_text(response.content)}

    # -------------------------
    # NODE 3: TOOL EXECUTION
    # -------------------------
    @traceable(name="tool")
    async def tool_node(state: State):
        tools = await get_tools()
        search_tool = tools[0]

        result = await search_tool.ainvoke({
            "query": state["search_query"]
        })

        # FIX: result itself may be a list, or an object with .content
        if isinstance(result, list):
            tool_text = extract_text(result)
        elif hasattr(result, "content"):
            tool_text = extract_text(result.content)
        else:
            tool_text = str(result)

        return {"tool_result": tool_text}

    # -------------------------
    # NODE 4: ANSWER GENERATION
    # -------------------------
    @traceable(name="answer")
    async def answer_node(state: State):
        messages = [SystemMessage(content=ANSWER_PROMPT)]

        if state.get("summary"):
            messages.append(
                SystemMessage(content=f"Conversation Summary: {state['summary']}")
            )

        messages.extend(state["messages"][-10:])
        messages.append(
            HumanMessage(content=f"Search Results:\n{state['tool_result']}")
        )

        response = await llm_answer.ainvoke(messages)

        # FIX: force content to string — Gemini can return list of blocks here too
        content = extract_text(response.content)

        return {
            "final_answer": content,
            "messages": [AIMessage(content=content)]
        }

    # -------------------------
    # NODE 5: REFUSAL
    # -------------------------
    async def refusal_node(state: State):
        msg = "I can only assist with software development and technical documentation queries."
        return {
            "final_answer": msg,
            "messages": [AIMessage(content=msg)]
        }

    # -------------------------
    # NODE 6: SUMMARIZE
    # Only runs when messages >= 40; never touches final_answer
    # -------------------------
    async def summarize_node(state: State):
        messages = state["messages"]

        if len(messages) < 40:
            return {}  # preserve all existing state untouched

        conversation = "\n".join(
            f"{msg.type}: {extract_text(msg.content)}" for msg in messages[:-10]
        )
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
            "summary": extract_text(response.content),
            "messages": messages[-10:]
            # final_answer intentionally NOT set here
        }

    # -------------------------
    # ROUTER
    # -------------------------
    def route_decision(state: State):
        return state["route"]

    # -------------------------
    # GRAPH ASSEMBLY
    # -------------------------
    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)
    graph.add_node("rewrite",    rewrite_node)
    graph.add_node("tool",       tool_node)
    graph.add_node("answer",     answer_node)
    graph.add_node("refusal",    refusal_node)
    graph.add_node("summarize",  summarize_node)

    graph.set_entry_point("classifier")

    graph.add_conditional_edges(
        "classifier",
        route_decision,
        {
            "ALLOW": "rewrite",
            "BLOCK": "refusal"
        }
    )

    graph.add_edge("rewrite",   "tool")
    graph.add_edge("tool",      "answer")
    graph.add_edge("answer",    "summarize")
    graph.add_edge("summarize", END)
    graph.add_edge("refusal",   END)

    return graph.compile(checkpointer=checkpointer)


# -------------------------
# RUN FUNCTION
# Fresh DB connection every call — safe across asyncio.run() boundaries
# -------------------------
def run_sync(query: str, thread_id: str):

    async def _run():
        DB_URI = os.getenv("DATABASE_URL")
        if not DB_URI:
            raise ValueError("DATABASE_URL not found in environment")

        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            app = await build_graph(checkpointer)

            config = {
                "configurable": {"thread_id": thread_id},
                "metadata":     {"chat_id":   thread_id}
            }

            result = await app.ainvoke(
                {
                    "query":    query,
                    "messages": [HumanMessage(content=query)]
                },
                config=config
            )

            # FIX: final_answer could be str, list, or missing — handle all cases
            raw_final = result.get("final_answer", "")
            final = extract_text(raw_final).strip()

            if final:
                return final

            # Fallback: last AI message in state
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage):
                    text = extract_text(msg.content).strip()
                    if text:
                        return text

            return "No response was generated."

    return asyncio.run(_run())