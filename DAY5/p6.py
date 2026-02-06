'''
 STAGE 2 — Two Agents (Researcher + Writer)
Goal:
Two agents collaborate:
Agent 1: Researcher
Uses web search
Extracts facts
Agent 2: Writer
Writes final response using research notes
'''
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# -----------------------
# State
# -----------------------
class State(TypedDict):
    question: str
    research_notes: str
    final_answer: str

# -----------------------
# LLM + Tool
# -----------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
search = DuckDuckGoSearchRun()

# -----------------------
# Agent 1: Researcher
# -----------------------
def researcher_agent(state: State):
    query = state["question"]
    web = search.run(query)

    prompt = f"""
You are a researcher agent.
Extract important facts from the web results.

Question: {query}

Web Results:
{web}

Return bullet point notes only.
"""
    notes = llm.invoke(prompt).content
    return {"research_notes": notes}

# -----------------------
# Agent 2: Writer
# -----------------------
def writer_agent(state: State):
    prompt = f"""
You are a writer agent.
Write a final clear answer using research notes.

Question: {state['question']}

Research Notes:
{state['research_notes']}

Write final answer in 3-5 lines.
"""
    ans = llm.invoke(prompt).content
    return {"final_answer": ans}

# -----------------------
# Graph
# -----------------------
graph = StateGraph(State)

graph.add_node("researcher", researcher_agent)
graph.add_node("writer", writer_agent)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", END)

app = graph.compile()

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    out = app.invoke({"question": "Explain what is LangGraph in simple terms.", "research_notes": "", "final_answer": ""})
    print("\nFINAL ANSWER:\n", out["final_answer"])
