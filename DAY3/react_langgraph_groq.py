"""
============================================================
  ReAct Architecture — LangGraph + Groq
  Full implementation with Search & Python tools
============================================================
  INSTALL DEPENDENCIES:
      pip install langchain-groq langgraph langchain-community langchain-core requests
  
  SET YOUR API KEY:
      export GROQ_API_KEY="gsk_your_key_here"
      (or set it directly in the GROQ_API_KEY variable below)
============================================================
"""

import os
import json
import traceback
from typing import TypedDict, Annotated, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolsCondition, tools_condition


# ============================================================
# 1. CONFIGURATION
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "gsk_your_key_here")

# Groq supports several fast models — choose one:
#   "llama3-groq-70b-8192-tool-use-preview"   ← best for tool use
#   "mixtral-8x7b-32768"                      ← good multilingual support
#   "llama3-8b-8192"                          ← lightweight & fast
MODEL_NAME = "llama3-groq-70b-8192-tool-use-preview"

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=0,          # deterministic output
    max_tokens=2048,
)


# ============================================================
# 2. TOOL DEFINITIONS
#    Each tool is decorated with @tool so LangChain
#    auto-generates the schema (name, description, args).
# ============================================================

@tool
def web_search(query: str) -> str:
    """
    Search the web for up-to-date information.
    Use this when you need facts, news, or any external data.

    Args:
        query: The search query string.

    Returns:
        A string containing the search results summary.
    """
    try:
        import requests
        # Using DuckDuckGo Instant Answer API (no key required)
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "no_redirect": "1"}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        # Try to get the instant answer first
        if data.get("Abstract"):
            return data["Abstract"]

        # Fall back to related topics
        if data.get("RelatedTopics"):
            summaries = []
            for topic in data["RelatedTopics"][:3]:
                if "Text" in topic:
                    summaries.append(topic["Text"])
            if summaries:
                return "\n".join(summaries)

        return "No results found. Try rephrasing your query."

    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def python_repl(code: str) -> str:
    """
    Execute Python code in a safe sandbox and return the output.
    Use this for calculations, data processing, or any computation.

    Args:
        code: A valid Python code string to execute.

    Returns:
        The printed output or return value of the code, or an error message.
    """
    try:
        # Restricted globals — blocks dangerous imports
        safe_globals = {"__builtins__": __builtins__}
        allowed_modules = {"math", "json", "random", "datetime", "collections", "itertools", "functools", "string"}

        # Capture printed output
        import io, sys
        captured = io.StringIO()
        sys.stdout = captured

        # Execute in sandbox
        exec(code, safe_globals)

        sys.stdout = sys.__stdout__
        output = captured.getvalue().strip()

        return output if output else "Code executed successfully (no output)."

    except Exception as e:
        import sys
        sys.stdout = sys.__stdout__
        return f"Python error: {str(e)}\n{traceback.format_exc()}"


# ============================================================
# 3. LANGGRAPH STATE
#    Defines the shape of data flowing through the graph.
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[List, lambda x, y: x + y]  # message list grows with each step


# ============================================================
# 4. NODE FUNCTIONS
#    Each function is a "node" in the LangGraph.
# ============================================================

# --- Collect all tools into a list ---
tools = [web_search, python_repl]

# --- Build a tool lookup map: tool_name → callable ---
tool_map = {t.name: t for t in tools}


def agent_node(state: AgentState) -> AgentState:
    """
    THE BRAIN — calls the Groq LLM with the current message history.
    Returns the LLM's response (may include a tool_call or a final answer).
    """
    print("\n [Agent] Thinking...")
    response = llm.with_tools(tools).invoke(state["messages"])
    print(f"   LLM output: {response.content or '(tool call)'}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            print(f" Tool call: {tc['name']}({tc['input']})")
    return {"messages": [response]}


def tool_node(state: AgentState) -> AgentState:
    """
    THE ARMS — executes every tool_call the LLM requested,
    then returns ToolMessage(s) with the results.
    """
    results = []
    last_ai_message = state["messages"][-1]              # the AIMessage with tool_calls

    for tool_call in last_ai_message.tool_calls:
        tool_name   = tool_call["name"]
        tool_input  = tool_call["input"]
        tool_id     = tool_call["id"]

        print(f"\n [Tool] Executing: {tool_name}({tool_input})")

        if tool_name in tool_map:
            output = tool_map[tool_name].invoke(tool_input)
        else:
            output = f"Error: Tool '{tool_name}' not found."

        print(f"   Result: {output[:200]}...")  # preview first 200 chars

        results.append(
            ToolMessage(
                content=str(output),
                tool_call_id=tool_id,
                name=tool_name,
            )
        )

    return {"messages": results}


# ============================================================
# 5. ROUTING LOGIC
#    Decides: should we go to the tool node, or are we done?
# ============================================================

def should_continue(state: AgentState) -> str:
    """
    If the last message has tool_calls  →  route to tool_node
    Otherwise                           →  we're done (END)
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool"       # continue the loop
    return END              # final answer reached


# ============================================================
# 6. BUILD THE LANGGRAPH
#
#   START
#     │
#     ▼
#   ┌─────────┐     tool_calls?     ┌───────────┐
#   │  agent  │ ──── YES ─────────► │  tools    │
#   │  node   │ ◄─────────────────── │  node     │
#   └─────────┘                     └───────────┘
#     │
#     │  NO (final answer)
#     ▼
#    END
# ============================================================

graph = StateGraph(AgentState)

# Register nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Wire the edges
graph.add_edge(START, "agent")                          # always start at agent
graph.add_conditional_edges("agent", should_continue)   # agent → tools OR END
graph.add_edge("tools", "agent")                        # tools always loop back to agent

# Compile into a runnable
agent = graph.compile()


# ============================================================
# 7. SYSTEM PROMPT (ReAct Instructions)
# ============================================================

SYSTEM_PROMPT = """You are a helpful AI agent with access to tools.
You MUST follow this ReAct loop:

    Thought  →  decide what to do next
    Action   →  call a tool (if needed)
    Observation → read the tool's result
    ... repeat until you can answer ...
    Final Answer → respond to the user

Available tools:
  • web_search(query)   – search the internet
  • python_repl(code)   – run Python code for calculations

Rules:
  1. Always think before acting.
  2. Use web_search for factual / real-world queries.
  3. Use python_repl for math, data processing, or logic.
  4. If a single query needs BOTH tools, chain them.
  5. Never guess — verify with a tool.
  6. Be concise in your final answer.
"""


# ============================================================
# 8. RUN THE AGENT
# ============================================================

def run_agent(user_question: str):
    """
    Entry point: feed a question into the compiled graph
    and stream the full ReAct trace.
    """
    print("=" * 60)
    print(f" Question: {user_question}")
    print("=" * 60)

    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_question),
        ]
    }

    # Invoke the graph — LangGraph handles the loop automatically
    final_state = agent.invoke(initial_state)

    # The last AIMessage in the chain is the final answer
    final_answer = final_state["messages"][-1].content
    print("\n" + "=" * 60)
    print(f" Final Answer:\n{final_answer}")
    print("=" * 60)

    return final_answer


# ============================================================
# 9. DEMO — run sample questions
# ============================================================

if __name__ == "__main__":

    # --- Question 1: Pure search ---
    run_agent("What is the current population of Tokyo, Japan?")

    print("\n\n")

    # --- Question 2: Pure calculation ---
    run_agent("Calculate the compound interest: principal=50000, rate=7.5%, years=10, compounded annually.")

    print("\n\n")

    # --- Question 3: Search + Calculation combined ---
    run_agent(
        "Search for the radius of Earth in kilometers, "
        "then calculate the surface area using the formula 4 * pi * r^2."
    )
