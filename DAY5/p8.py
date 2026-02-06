'''
the REAL multi-agent loop in LangGraph (Agentic AI style), where:

Researcher gathers info
Writer writes draft
Reviewer evaluates quality
If quality is low → sends back to Writer for improvement (loop)
Stops when quality is good OR max iterations reached

This is the true Agentic AI loop (not just a linear pipeline).
'''
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun


# -----------------------
# 1) Define State
# -----------------------
class State(TypedDict):
    question: str
    research_notes: str
    draft_answer: str
    reviewed_answer: str
    feedback: str
    score: int
    iteration: int


# -----------------------
# 2) LLM + Tool
# -----------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
search = DuckDuckGoSearchRun()


# -----------------------
# 3) Agents (Nodes)
# -----------------------

def researcher_agent(state: State):
    """Agent 1: Researcher - collect facts from web"""
    web = search.run(state["question"])

    prompt = f"""
You are a Researcher agent.
Extract key facts from web results.

Question: {state['question']}

Web Results:
{web}

Return bullet point research notes only.
"""
    notes = llm.invoke(prompt).content
    return {"research_notes": notes}


def writer_agent(state: State):
    """Agent 2: Writer - produce draft, and improve using reviewer feedback"""
    feedback = state.get("feedback", "")
    iteration = state.get("iteration", 0)

    prompt = f"""
You are a Writer agent.

Write the best possible answer.

Question:
{state['question']}

Research Notes:
{state['research_notes']}

Reviewer Feedback (if any):
{feedback}

Write a clear, concise answer with correct facts.
"""
    draft = llm.invoke(prompt).content
    return {
        "draft_answer": draft,
        "iteration": iteration + 1
    }


def reviewer_agent(state: State):
    """Agent 3: Reviewer - score answer and provide feedback"""

    prompt = f"""
You are a Reviewer agent.
Evaluate the draft answer.

Question:
{state['question']}

Draft Answer:
{state['draft_answer']}

Tasks:
1) Give a score from 1 to 10 (10 = excellent).
2) Provide short feedback to improve the answer.
3) Provide an improved final answer.

Return strictly in this format:
SCORE: <number>
FEEDBACK: <text>
IMPROVED: <text>
"""
    resp = llm.invoke(prompt).content

    # Simple parsing
    score = 0
    feedback = ""
    improved = state["draft_answer"]

    for line in resp.splitlines():
        if line.startswith("SCORE:"):
            try:
                score = int(line.replace("SCORE:", "").strip())
            except:
                score = 0
        elif line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()
        elif line.startswith("IMPROVED:"):
            improved = line.replace("IMPROVED:", "").strip()

    # Fallback if model didn't follow format perfectly
    if score == 0:
        score = 7  # assume medium
        feedback = "Improve clarity and ensure the answer directly addresses the question."
        improved = resp

    return {
        "score": score,
        "feedback": feedback,
        "reviewed_answer": improved
    }


# -----------------------
# 4) Routing Logic (Loop Decision)
# -----------------------
def should_continue(state: State):
    """
    If score is good enough OR iterations exceeded -> stop.
    Else -> go back to writer.
    """
    score = state.get("score", 0)
    iteration = state.get("iteration", 0)

    if score >= 8:
        return "end"
    if iteration >= 3:
        return "end"
    return "rewrite"


# -----------------------
# 5) Build Graph
# -----------------------
graph = StateGraph(State)

graph.add_node("researcher", researcher_agent)
graph.add_node("writer", writer_agent)
graph.add_node("reviewer", reviewer_agent)

graph.set_entry_point("researcher")

# Normal flow
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")

# Conditional loop after review
graph.add_conditional_edges(
    "reviewer",
    should_continue,
    {
        "rewrite": "writer",
        "end": END
    }
)

app = graph.compile()


# -----------------------
# 6) Run
# -----------------------
if __name__ == "__main__":
    query = "What is Agentic AI? Explain with one real-world example."

    initial_state = {
        "question": query,
        "research_notes": "",
        "draft_answer": "",
        "reviewed_answer": "",
        "feedback": "",
        "score": 0,
        "iteration": 0
    }

    output = app.invoke(initial_state)

    print("\n==============================")
    print("FINAL OUTPUT")
    print("==============================")
    print("Iterations:", output["iteration"])
    print("Final Score:", output["score"])
    print("\nFinal Answer:\n", output)
    print("\nFinal Answer:\n", output["research_notes"])


from IPython.display import Image,display

#display(Image(app.get_graph().draw_mermaid_png()))
png_bytes = app.get_graph().draw_mermaid_png()

with open("agentLoop.png", "wb") as f:
    f.write(png_bytes)

print("Saved agenLoop.png")
'''
This graph will behave like:

Researcher collects facts

Writer writes draft

Reviewer scores it

If score < 8 → Reviewer gives feedback

Writer rewrites using feedback

Reviewer rechecks

Stops when good

That is a real agentic loop.
'''