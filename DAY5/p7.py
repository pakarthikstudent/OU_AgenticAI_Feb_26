'''
 STAGE 3 — Three Agents (Researcher + Writer + Reviewer)
Goal:
Three agents:
Agent 1: Researcher
search + notes
Agent 2: Writer
creates draft answer
Agent 3: Reviewer
checks correctness + improves
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
    draft_answer: str
    reviewed_answer: str

# -----------------------
# LLM + Tool
# -----------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
search = DuckDuckGoSearchRun()

# -----------------------
# Agent 1: Researcher
# -----------------------
def researcher_agent(state: State):
    web = search.run(state["question"])
    prompt = f"""
You are a researcher agent.
Extract key facts from web results.

Question: {state['question']}

Web Results:
{web}

Return bullet points.
"""
    notes = llm.invoke(prompt).content
    return {"research_notes": notes}

# -----------------------
# Agent 2: Writer
# -----------------------
def writer_agent(state: State):
    prompt = f"""
You are a writer agent.
Write a clear draft answer using notes.

Question: {state['question']}

Notes:
{state['research_notes']}

Write a helpful answer.
"""
    draft = llm.invoke(prompt).content
    return {"draft_answer": draft}

# -----------------------
# Agent 3: Reviewer
# -----------------------
def reviewer_agent(state: State):
    prompt = f"""
You are a reviewer agent.
Improve the draft answer.
- fix incorrect info
- make it concise
- ensure it answers the question

Question: {state['question']}

Draft:
{state['draft_answer']}

Return improved final answer.
"""
    reviewed = llm.invoke(prompt).content
    return {"reviewed_answer": reviewed}

# -----------------------
# Graph
# -----------------------
graph = StateGraph(State)

graph.add_node("researcher", researcher_agent)
graph.add_node("writer", writer_agent)
graph.add_node("reviewer", reviewer_agent)

graph.set_entry_point("researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_edge("reviewer", END)

app = graph.compile()

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    out = app.invoke({
        "question": "What is Agentic AI? Explain with a real-life example.",
        "research_notes": "",
        "draft_answer": "",
        "reviewed_answer": ""
    })
    print("\nFINAL REVIEWED ANSWER:\n", out["reviewed_answer"])
