import os
from typing import TypedDict

# Let's create the state first

class pipelinestate(TypedDict):
  raw_input: str
  edited_text: str
  script_text: str
  final_output: str

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm= ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

# Editor Node
def editor_node(state: pipelinestate) -> dict:
  """Stage 1. Cleans up grammar, removes typos,and refines the tone"""

  prompt = (
    "You are a professional copyeditor. Clean up the following raw text." "You have to clean up grammar, remove typos, spelling mistakes and refine the tone of the following text smooth it out while keeping the meaning intact."
    "Return only the edited text \n\n"
    f"Text:\n{state['raw_input']}"
  )

  response = llm.invoke(prompt)
  return {"edited_text": response.content.strip()}

# Script Writer Node
def scriptwiter_node(state: pipelinestate) -> dict:
  """Stage 2: Formats the clean text into an enagaging video script style."""
  print("\n--- [Stage 2] Executing Script Writer Node ---\n")

  prompt=(
    "You are a charismatic Youtube content creator. Take this edited text and transform it into a highly engaging, punchy, conversationa video script hook."
    "Make it sound like a real person speaking passsionately."
    "Return only the script content \n\n"
    f"Edited Text:\n{state['edited_text']}"
  )

  response = llm.invoke(prompt)
  return {"script_text": response.content.strip()}

# Translator Node
def translator_node(state: pipelinestate) -> dict:
  """Stage 3: Translates the script into a natural flowing Hinglish."""
  print("\n--- [Stage 3] Executing Hinglish Translator Node ---\n")

  prompt=(
    "You are an expert content localizer for the Indian market. Take the following script "
    "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
    "or repeat information. Alternating comfortably between Hindi and English phrases just like "
    "an intellectual tech educator would speak naturally on a live stream. Keep the energy high! "
    "Return only the final Hinglish text.\n\n"
    f"Script:\n{state['script_text']}"
  )

  response = llm.invoke(prompt)
  return {"final_output": response.content.strip()}

# Now state and nodes are ready and now we can create the graph
# For creating the graph you have to connect this node and for that you have to use the edges
# edges are very important to create the workflows

from langgraph.graph import StateGraph, START, END

# Create the Graph

graph = StateGraph(pipelinestate)

#add the nodes in our graph 

graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwiter_node)
graph.add_node("translator", translator_node)

#Add edges (sequential - one after another)

graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)

# Compile the graph
app = graph.compile()

result = app.invoke({
    "raw_input" :"AI agents are the future of tech. They can think, plan, and act on their own. LangGraph helps you build these agents with proper control and memory."
})

#output 
print("your result are : - \n\n")
print(result['final_output'])