# chain_based_agentic_langgraph
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun


# ----------------------------
# 1) Define State
# ----------------------------
class ChainState(TypedDict):
    user_question: str
    plan: str
    web_result: str
    draft_answer: str
    final_answer: str


# ----------------------------
# 2) LLM + Tool
# ----------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
search = DuckDuckGoSearchRun()


# ----------------------------
# 3) Chain Nodes
# ----------------------------

def planner_node(state: ChainState):
    """Chain Step 1: Create plan"""
    prompt = f"""
You are a planning assistant.
Create a short step-by-step plan to answer the question.

Question: {state['user_question']}

Return plan in 3 bullet points.
"""
    plan = llm.invoke(prompt).content
    return {"plan": plan}


def research_node(state: ChainState):
    """Chain Step 2: Tool usage (search)"""
    query = state["user_question"]
    web_result = search.run(query)
    return {"web_result": web_result}


def writer_node(state: ChainState):
    """Chain Step 3: Generate draft using plan + web result"""
    prompt = f"""
You are a writer assistant.

User Question:
{state['user_question']}

Plan:
{state['plan']}

Web Result:
{state['web_result']}

Write a clear draft answer.
"""
    draft = llm.invoke(prompt).content
    return {"draft_answer": draft}


def formatter_node(state: ChainState):
    """Chain Step 4: Format output"""
    prompt = f"""
Format the following answer in a clean structure:

- Title
- Key Points (3 bullets)
- Final Summary (2 lines)

Draft:
{state['draft_answer']}
"""
    final = llm.invoke(prompt).content
    return {"final_answer": final}


# ----------------------------
# 4) Build Graph (Chain)
# ----------------------------
graph = StateGraph(ChainState)

graph.add_node("planner", planner_node)
graph.add_node("research", research_node)
graph.add_node("writer", writer_node)
graph.add_node("formatter", formatter_node)

graph.set_entry_point("planner")

graph.add_edge("planner", "research")
graph.add_edge("research", "writer")
graph.add_edge("writer", "formatter")
graph.add_edge("formatter", END)

app = graph.compile()


# ----------------------------
# 5) Run
# ----------------------------
if __name__ == "__main__":
    question = "What is Agentic AI? Explain with one real-world example."

    output = app.invoke({
        "user_question": question,
        "plan": "",
        "web_result": "",
        "draft_answer": "",
        "final_answer": ""
    })

    print("\n================ FINAL ANSWER ================\n")
    print(output["final_answer"])

from IPython.display import Image,display

#display(Image(app.get_graph().draw_mermaid_png()))
png_bytes = app.get_graph().draw_mermaid_png()

with open("chain_based_loop.png", "wb") as f:
    f.write(png_bytes)

print("Saved chain_based_loop.png")