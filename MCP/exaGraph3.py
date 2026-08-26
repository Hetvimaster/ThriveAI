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
from langgraph.store.postgres.aio import AsyncPostgresStore
import json

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
    # user_id is passed in so nodes can read/write to the store
    user_id: str


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

MEMORY_EXTRACT_PROMPT = """
You are a memory extraction assistant.

Given a conversation exchange, extract any useful long-term facts about the user.

Extract ONLY concrete facts worth remembering across sessions, such as:
- preferred programming language
- experience level (beginner / intermediate / expert)
- current project they are working on
- frameworks or tools they use
- specific preferences (e.g. "prefers concise answers", "wants code examples")
- their name if mentioned

Return a JSON object with short keys and string values.
If nothing meaningful is found, return an empty JSON object: {}

Return ONLY valid JSON. No explanation. No markdown. No extra text.

Example output:
{
  "preferred_language": "Python",
  "experience_level": "intermediate",
  "current_project": "LangGraph MCP agent",
  "prefers": "concise answers with code examples"
}
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
async def build_graph(checkpointer, store):
    # store is AsyncPostgresStore — used for cross-thread user memory

    llm_router   = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    llm_rewriter = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    llm_answer   = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

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

        if isinstance(result, list):
            tool_text = extract_text(result)
        elif hasattr(result, "content"):
            tool_text = extract_text(result.content)
        else:
            tool_text = str(result)

        return {"tool_result": tool_text}

    # -------------------------
    # NODE 4: ANSWER GENERATION
    # Reads cross-thread user memory from store and injects into prompt
    # -------------------------
    @traceable(name="answer")
    async def answer_node(state: State):
        messages = [SystemMessage(content=ANSWER_PROMPT)]

        # ── CROSS-THREAD MEMORY ──────────────────────────────────────────
        # Fetch user facts stored across ALL chat threads for this user.
        # Namespace: ("user_memory", user_id) — same regardless of thread.
        # This is what makes the AI "know" the user even in a brand new chat.
        user_id = state.get("user_id", "")
        if user_id:
            try:
                memory_items = await store.asearch(("user_memory", user_id))
                if memory_items:
                    # Merge all stored fact dicts into one block
                    all_facts = {}
                    for item in memory_items:
                        if isinstance(item.value, dict):
                            all_facts.update(item.value)

                    if all_facts:
                        facts_text = "\n".join(
                            f"- {k}: {v}" for k, v in all_facts.items()
                        )
                        messages.append(SystemMessage(content=f"""
Long-term memory about this user (learned from past conversations):
{facts_text}

Use this to personalize your response where relevant.
Do NOT mention that you have this memory unless asked.
                        """))
            except Exception:
                pass  # if store read fails, continue without memory
        # ────────────────────────────────────────────────────────────────

        # Within-thread summary (from summarize_node)
        if state.get("summary"):
            messages.append(
                SystemMessage(content=f"Conversation Summary: {state['summary']}")
            )

        messages.extend(state["messages"][-10:])
        messages.append(
            HumanMessage(content=f"Search Results:\n{state['tool_result']}")
        )

        response = await llm_answer.ainvoke(messages)
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
    # Within-thread summarization — trims long message lists
    # -------------------------
    async def summarize_node(state: State):
        messages = state["messages"]

        if len(messages) < 40:
            return {}

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
        }

    # -------------------------
    # NODE 7: MEMORY EXTRACTION
    # Runs after every answer — extracts user facts and writes to store.
    # These facts persist across ALL chat threads for this user.
    # -------------------------
    async def memory_node(state: State):
        """
        Extract long-term facts from the latest exchange and save to store.

        HOW IT WORKS:
        - Takes the last human message and last AI response
        - Asks LLM to extract any facts worth remembering
        - Saves them to AsyncPostgresStore under ("user_memory", user_id)
        - Next time this user sends ANY message (even new chat), answer_node
          reads these facts and personalizes the response

        NAMESPACE DESIGN:
        ("user_memory", user_id) — scoped to the user, NOT the thread
        So "a3f9b2c1" always reads/writes the same memory bucket
        regardless of whether they're in chat _0, _1, or _99
        """
        user_id = state.get("user_id", "")
        if not user_id:
            return {}

        messages = state["messages"]

        # Get last human + AI exchange
        last_human = next(
            (extract_text(m.content) for m in reversed(messages)
             if isinstance(m, HumanMessage)), ""
        )
        last_ai = next(
            (extract_text(m.content) for m in reversed(messages)
             if isinstance(m, AIMessage)), ""
        )

        if not last_human or not last_ai:
            return {}

        extract_prompt = f"""
{MEMORY_EXTRACT_PROMPT}

Conversation:
User: {last_human}
Assistant: {last_ai}
"""
        try:
            response = await llm_answer.ainvoke(extract_prompt)
            raw = extract_text(response.content).strip()

            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()

            facts = json.loads(raw)

            if facts and isinstance(facts, dict):
                # Write to store — merges with existing facts
                # Key: "preferences" — you can use multiple keys for different
                # categories (e.g. "project", "style", "tools")
                await store.aput(
                    ("user_memory", user_id),
                    "preferences",
                    facts
                )
        except Exception:
            pass  # if extraction fails, silently continue

        return {}

    # -------------------------
    # ROUTER
    # -------------------------
    def route_decision(state: State):
        return state["route"]

    # -------------------------
    # GRAPH ASSEMBLY
    #
    # Flow:
    # classifier → rewrite → tool → answer → memory_node → summarize → END
    #           ↘ refusal → END
    #
    # memory_node runs after every answer so facts are always up to date
    # summarize_node runs after memory so it has the latest messages
    # -------------------------
    graph = StateGraph(State)

    graph.add_node("classifier",  classifier_node)
    graph.add_node("rewrite",     rewrite_node)
    graph.add_node("tool",        tool_node)
    graph.add_node("answer",      answer_node)
    graph.add_node("refusal",     refusal_node)
    graph.add_node("memory",      memory_node)
    graph.add_node("summarize",   summarize_node)

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
    graph.add_edge("answer",    "memory")     # extract facts after every answer
    graph.add_edge("memory",    "summarize")  # then trim if needed
    graph.add_edge("summarize", END)
    graph.add_edge("refusal",   END)

    return graph.compile(checkpointer=checkpointer, store=store)


# -------------------------
# RUN FUNCTION
# Both checkpointer and store use the same Postgres DB
# checkpointer → langgraph_checkpoints table (per-thread history)
# store        → langgraph_store table (cross-thread user facts)
# -------------------------
def run_sync(query: str, thread_id: str, user_id: str):

    async def _run():
        DB_URI = os.getenv("DATABASE_URL")
        if not DB_URI:
            raise ValueError("DATABASE_URL not found in environment")

        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            async with AsyncPostgresStore.from_conn_string(DB_URI) as store:
                await checkpointer.setup()
                await store.setup()

                app = await build_graph(checkpointer, store)

                config = {
                    "configurable": {"thread_id": thread_id},
                    "metadata":     {"chat_id":   thread_id}
                }

                result = await app.ainvoke(
                    {
                        "query":    query,
                        "messages": [HumanMessage(content=query)],
                        "user_id":  user_id,   # passed so nodes can access store
                    },
                    config=config
                )

                raw_final = result.get("final_answer", "")
                final = extract_text(raw_final).strip()

                if final:
                    return final

                for msg in reversed(result.get("messages", [])):
                    if isinstance(msg, AIMessage):
                        text = extract_text(msg.content).strip()
                        if text:
                            return text

                return "No response was generated."

    return asyncio.run(_run())
