# LangGraph + Groq ReAct Agent - Jupyter Notebook

"""
!pip install langgraph langchain langchain-core langchain-groq
"""

import os
from typing import TypedDict, Annotated, List, Tuple, Union
import operator

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_groq import ChatGroq
from langchain_core.tools import tool

print(" All imports successful!")

# Get free key at: https://console.groq.com
os.environ['GROQ_API_KEY'] = 'your-groq-api-key-here'  #  Replace this!

# Verify it's set
if os.getenv('GROQ_API_KEY') and os.getenv('GROQ_API_KEY') != 'your-groq-api-key-here':
    print(" API key set!")
else:
    print("  Please set your GROQ_API_KEY above!")

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Simulated search results
    search_db = {
        "tokyo weather april": "Tokyo in April: 14-20°C (57-68°F). Cherry blossom season peaks early-mid April.",
        "tokyo activities april": "Top activities: Hanami at Ueno Park, Sumida River, temple visits, sakura foods.",
        "paris hotels": "Budget hotels: Hotel du Nord (€120), Ibis Paris (€110).",
    }
    
    query_lower = query.lower()
    for key, result in search_db.items():
        if all(word in query_lower for word in key.split()):
            return result
    
    return f"Search results for '{query}'"

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def weather_forecast(city: str) -> str:
    """Get weather forecast for a city."""
    forecasts = {
        "tokyo": "Tokyo: Partly cloudy, 18°C, Humidity: 65%",
        "paris": "Paris: Light rain, 12°C, Humidity: 78%",
        "new york": "New York: Sunny, 22°C, Humidity: 55%"
    }
    
    for key, forecast in forecasts.items():
        if key in city.lower():
            return forecast
    
    return f"Weather for {city}: Moderate conditions, ~20°C"

tools = [web_search, calculator, weather_forecast]
tool_executor = ToolExecutor(tools)

print(f" Defined {len(tools)} tools: {[t.name for t in tools]}")

class AgentState(TypedDict):
    """State that flows through the graph."""
    messages: Annotated[List[BaseMessage], operator.add]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[Tuple[AgentAction, str]], operator.add]

print("State schema defined!")

def run_agent(state: AgentState) -> AgentState:
    """Agent node - LLM reasoning and decision making."""
    print("\n AGENT: Thinking...")
    
    # Initialize LLM
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    # Get messages
    messages = state.get("messages", [])
    
    # LLM generates response
    response = llm_with_tools.invoke(messages)
    
    # Check if using tools
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"    Decision: Use tool '{tool_call['name']}'")
        
        return {
            "messages": [response],
            "agent_outcome": AgentAction(
                tool=tool_call['name'],
                tool_input=tool_call['args'],
                log=""
            )
        }
    else:
        print(f"    Decision: Finish with answer")
        return {
            "messages": [response],
            "agent_outcome": AgentFinish(
                return_values={"output": response.content},
                log=""
            )
        }

print("Agent node defined!")

def execute_tools(state: AgentState) -> AgentState:
    """Tools node - Execute the action."""
    agent_action = state["agent_outcome"]
    
    print(f"\n🔧 TOOLS: Executing {agent_action.tool}")
    
    # Execute tool
    observation = tool_executor.invoke(agent_action)
    
    print(f" Result: {str(observation)[:80]}...")
    
    # Create tool message
    tool_message = ToolMessage(
        content=str(observation),
        tool_call_id=state["messages"][-1].tool_calls[0]['id']
    )
    
    return {
        "messages": [tool_message],
        "intermediate_steps": [(agent_action, str(observation))]
    }

print("Tools node defined!")

# ============================================================================
#  Routing Logic
# ============================================================================
def should_continue(state: AgentState) -> str:
    """Decide whether to continue or end."""
    if isinstance(state["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"

print("Routing logic defined!")

# ============================================================================
#  Build the Graph
# ============================================================================
def create_agent():
    """Build and compile the graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", run_agent)
    workflow.add_node("tools", execute_tools)
    
    # Set entry
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    # Compile
    return workflow.compile()

print(" Graph builder ready!")

# ============================================================================
# : Run Agent Function
# ============================================================================
def run_query(query: str):
    """Run a query through the agent."""
    print("\n" + "="*70)
    print(f" QUERY: {query}")
    print("="*70)
    
    # Create agent
    app = create_agent()
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "intermediate_steps": []
    }
    
    # Execute
    final_state = app.invoke(initial_state)
    
    # Get answer
    if isinstance(final_state["agent_outcome"], AgentFinish):
        answer = final_state["agent_outcome"].return_values["output"]
    else:
        answer = "Agent did not complete"
    
    # Print summary
    print("\n" + "-"*70)
    print(f" ANSWER:\n{answer}")
    print("-"*70)
    print(f" Tools used: {len(final_state.get('intermediate_steps', []))}")
    print(f" Total messages: {len(final_state['messages'])}")
    print("="*70 + "\n")
    
    return answer

print(" Run function ready!")

# ============================================================================
# : Test Query 1 - Tokyo Travel
# ============================================================================
run_query("I want to visit Tokyo in April. What's the weather like and suggest some activities.")

# ============================================================================
# : Test Query 2 - Math
# ============================================================================
run_query("Calculate 234 multiplied by 567")

# ============================================================================
# : Test Query 3 - Weather
# ============================================================================
run_query("What's the weather forecast for New York?")

# ============================================================================
#  Test Query 4 - Your Own Question
# ============================================================================
# Try your own question here!
run_query("Your question here")

# ============================================================================
#  Stream Execution (See Each Step)
# ============================================================================
def run_with_streaming(query: str):
    """Run query and stream each step."""
    print(f"\n{'='*70}")
    print(f"STREAMING EXECUTION: {query}")
    print('='*70 + "\n")
    
    app = create_agent()
    
    for i, step in enumerate(app.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="values"
    ), 1):
        print(f"\n--- Step {i} ---")
        if "messages" in step and step["messages"]:
            last_msg = step["messages"][-1]
            msg_type = type(last_msg).__name__
            print(f"Message Type: {msg_type}")
            
            if hasattr(last_msg, 'content') and last_msg.content:
                print(f"Content: {last_msg.content[:100]}...")
            
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"Tool Call: {tc['name']}({tc['args']})")

# Run with streaming
run_with_streaming("What's the weather in Tokyo?")

# ============================================================================
#  Interactive Mode (Optional)
# ============================================================================
def interactive_mode():
    """Ask multiple questions interactively."""
    print("\n INTERACTIVE MODE")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print(" Goodbye!")
            break
        
        if query:
            try:
                run_query(query)
            except Exception as e:
                print(f" Error: {e}\n")

# Uncomment to run:
# interactive_mode()

# ============================================================================
#  Quick Test Function
# ============================================================================
def quick_test():
    """Run all test scenarios."""
    scenarios = [
        "What's 15 * 234?",
        "Search for Tokyo weather in April",
        "Weather forecast for Paris",
        "Tokyo activities in April"
    ]
    
    for query in scenarios:
        try:
            run_query(query)
        except Exception as e:
            print(f"Error on '{query}': {e}\n")

# Run quick test
quick_test()

# ============================================================================
#  Simplified Version (Alternative)
# ============================================================================
# If you want the simplest possible version:

from langgraph.prebuilt import create_react_agent

def simple_version(query: str):
    """Simplest possible agent."""
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    agent = create_react_agent(llm, tools)
    
    result = agent.invoke({
        "messages": [HumanMessage(content=query)]
    })
    
    print(f"\nQuery: {query}")
    print(f"Answer: {result['messages'][-1].content}\n")

# Test it
simple_version("Calculate 100 * 200")
