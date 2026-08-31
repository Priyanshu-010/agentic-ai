import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# STEP 1 - Building the rag retriever

embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_retriver(pdf_path:str):
  loader = PyPDFLoader(pdf_path)
  document = loader.load()

  splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

  chunks = splitter.split_documents(document)

  vectorstore= FAISS.from_documents(chunks, embeddings)

  return vectorstore.as_retriever(search_kwargs={"k":4})

academic_retriever = build_retriver("academics_handbook.pdf")
fee_retriever= build_retriver("fee_structure.pdf")


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5)

# Step 2 - State

class State(TypedDict):
  programme: str
  messages: Annotated[list, add_messages]
  query_type: str
  retrieved_context: str

# STEP 3 - Nodes generation

def classifier_node(state:State)->dict:
  """Look at the latest user message and decide which path to take."""

  last_message = state['messages'][-1].content

  prompt = (
    "Classify the following student query into exactly one category: "
    "'academic', 'fee', or 'general'.\n\n"
    "Use 'academic' for questions about attendance, exams, grading, credits, "
    "promotion, course structure, summer training, or degree requirements.\n"
    "Use 'fee' for questions about tuition, payment, refund, late charges, "
    "scholarships, or any money-related topic.\n"
    "Use 'general' for greetings, casual talk, or anything not related to "
    "the college rules or fee.\n\n"
    f"Query: {last_message}\n\n"
    "Return only one word: academic, fee, or general."
  )

  response = llm.invoke(prompt)
  category = response.content.strip().lower()

  if "academic" in category:
    category = "academic"
  elif "fee" in category:
    category = "fee"
  else:
    category = "general"

  return {"query_type": category}

def academic_rag_node(state: State) -> dict:
  """Retrieves relevant chunks from the academics handbook."""
  query = state["messages"][-1].content
  docs = academic_retriever.invoke(query)
  context = "\n\n".join([doc.page_content for doc in docs])
  return {"retrieved_context": context}

def fee_rag_node(state: State) -> dict:
  """Retrieves relevant chunks from the fee structure PDF."""
  query = state["messages"][-1].content
  docs = fee_retriever.invoke(query)
  context = "\n\n".join([doc.page_content for doc in docs])
  return {"retrieved_context": context}


def general_node(state: State) -> dict:
  """Answers directly using the LLM's own knowledge, no retrieval needed."""
  return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

def response_node(state: State) -> dict:
  """Generates the final answer, personalized using the student's programme."""
  query = state["messages"][-1].content
  programme = state.get("programme", "Unknown")
  context = state["retrieved_context"]

  if context == "NO_RETRIEVAL_NEEDED":
      prompt = (
        f"You are a friendly college assistant talking to a {programme} student. "
        f"Answer this question using your own general knowledge:\n\n{query}"
      )
  else:
      prompt = (
        f"You are a college assistant helping a {programme} student. "
        f"Use the following context from the official college documents to answer "
        f"the question accurately. If the context mentions specific figures for "
        f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Give a clear, friendly, and precise answer."
      )

  response = llm.invoke(prompt)
  return {"messages": [("ai", response.content.strip())]}

# STEP 4 -  Router function

def route_query(state: State):
  if state['query_type'] == "academic":
    return "academic_rag"
  elif state['query_type'] == "fee":
    return "fee_rag"
  else:
    return "general"

# STEP 5 - Building the graph

graph = StateGraph(State)

# First parameter in add_node is the name of the state, second is the function

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

# Edges

graph.add_edge(START, "classifier")
graph.add_conditional_edges("classifier", route_query)
graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")
graph.add_edge("response", END)

app = graph.compile()

#step 6 - Run the code 

print("welcome to the College assistant \n\n")

print("which programe are you in ")
print("1. BCA")
print("2. BBA")
print("3. B.com (H)")

choice = input("\nEnter 1, 2 or 3 ")

programme_map = {
  "1": "BCA",
  "2": "BBA",
  "3": "B.Com (H)"
}
student_programme = programme_map.get(choice, "BCA")

print(f"\nGreat! You're set as a {student_programme} student.")

while True:
  user_query = input("You:  ")

  if user_query.lower() in ["exit","quit"]:
    break
  
  result = app.invoke({
    "programme": student_programme,
    "messages": [("human",user_query)]
  })

  print(f"Assistant : {result['messages'][-1].content}")


# Explanation of the code and topic
  """
  AI answer: 


Here's the full picture, explained as if you're seeing it for the first time:

The Big Idea
You're building a college chatbot that:

Knows which programme (BCA/BBA/B.Com) the student is in
Routes questions to the right knowledge source (academic handbook PDF, fee structure PDF, or general LLM knowledge)
Answers using only relevant info
LangGraph is the framework that lets you define this as a directed graph of nodes — each node is a step, edges are the flow between them.

The Imports (what each library does)
from langgraph.graph.message import add_messages  # reducer that APPENDS to the messages list
from langgraph.graph import StateGraph, START, END  # the graph builder + special entry/exit points
from langchain_groq import ChatGroq               # LLM client (hosted on Groq's API)
from langchain_community.document_loaders import PyPDFLoader  # reads a PDF into text
from langchain_text_splitters import RecursiveCharacterTextSplitter  # cuts text into chunks
from langchain_huggingface import HuggingFaceEmbeddings  # converts text → numeric vectors
from langchain_community.vectorstores import FAISS  # vector database (stores & searches vectors)
from dotenv import load_dotenv                    # loads API keys from a .env file

You don't need to memorize these. Just know each one does one job in the RAG pipeline.

STEP 1 — Building the RAG Retriever
This is the "brain" that answers questions from your PDFs. It has 4 sub-steps:

PDF → Load → Split → Embed → Store in FAISS → (later) Search

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

This downloads a small model that turns any text into a vector (a list of ~384 numbers). Text with similar meaning gets similar vectors. This is the key trick that makes "semantic search" possible.

def build_retriver(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

Reads the PDF and gives you a list of "documents" (each page or section as a text block).

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(document)

Splits the text into ~800-character pieces, with 100 characters of overlap between consecutive chunks. Why overlap? So a sentence that straddles a chunk boundary isn't lost.

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

FAISS.from_documents → converts every chunk into a vector using the embedding model and stores them in an in-memory index.
.as_retriever(search_kwargs={"k": 4}) → gives you a callable that, when you pass it a query string, returns the top 4 most similar chunks.
So academic_retriever and fee_retriever are just two search engines, one per PDF.

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5)

Your LLM. temperature=0.5 means "slightly creative but mostly factual" — good for a chatbot that should be accurate.

STEP 2 — The State
class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str

This is a shared blackboard that every node can read from and write to. Think of it as a global variable that LangGraph manages for you.

Key	What it holds	Who writes it
programme	"BCA" / "BBA" / "B.Com (H)"	Set once at the start
messages	Full conversation history	add_messages reducer appends new ones
query_type	"academic" / "fee" / "general"	classifier_node
retrieved_context	Chunks from PDF (or sentinel)	the RAG nodes

The Annotated[list, add_messages] part is important: it tells LangGraph to append to the list instead of replacing it. Without that, every new message would wipe the old ones.

STEP 3 — The Nodes (each is one "step" in the pipeline)
classifier_node
Reads the latest user message
Asks the LLM: "Is this academic, fee, or general?"
Returns {"query_type": "academic"} (or whichever)
LangGraph merges that into state automatically
academic_rag_node / fee_rag_node
No LLM call. Just queries the FAISS index.
Returns the top-4 chunks joined into one string → stored in retrieved_context
general_node
No retrieval. Returns the flag "NO_RETRIEVAL_NEEDED" so the next node knows to skip context.
response_node (the actual answerer)
This is where the LLM actually generates the reply:

If retrieved_context is the sentinel → prompt the LLM with no context (general knowledge)
Otherwise → stuff the retrieved chunks into the prompt as "Context:" and ask the LLM to answer based on them
Returns {"messages": [("ai", response.content)]} → this gets appended to the conversation history by the add_messages reducer
STEP 4 — The Router
def route_query(state: State):
    if state['query_type'] == "academic":
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"

This is a pure function — no side effects. It just looks at the state and returns the name of the next node to visit. LangGraph uses that string to decide which edge to follow.

STEP 5 — Wiring the Graph
graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

Register all 5 nodes. The first argument is the name (used in edges), the second is the function.

graph.add_edge(START, "classifier")

Entry point → always go to classifier first.

graph.add_conditional_edges("classifier", route_query)

After classifier, don't go to a fixed node — ask route_query which one to pick. The return value of that function must match a registered node name.

graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")

All three paths converge into the same response node.

graph.add_edge("response", END)

After answering, stop.

The resulting graph looks like:

START → classifier ──┬── academic_rag ──┐
                      ├── fee_rag ───────┼──→ response → END
                      └── general ───────┘

app = graph.compile()

compile() validates the graph (checks all edges point to real nodes, etc.) and produces a callable object you can invoke.

STEP 6 — Running It
while True:
    user_query = input("You:  ")
    if user_query.lower() in ["exit", "quit"]:
        break

    result = app.invoke({
        "programme": student_programme,
        "messages": [("human", user_query)]
    })

    print(f"Assistant: {result['messages'][-1].content}")

Each call to app.invoke(...):

Takes the input dict (programme + new human message)
Runs the graph: classifier → RAG node → response
Returns the final state — you grab the last message (the AI's reply)
One thing to note: each invoke call is stateless from LangGraph's perspective. The messages list you pass in only contains the current turn, not the full history. If you want multi-turn memory, you'd need to either:

Pass the full messages history each time, or
Use LangGraph's checkpointer (a built-in memory mechanism)
For a simple demo, what you have works fine.

TL;DR — The Data Flow
User types "What's the late fee for BCA?"
        ↓
classifier_node: LLM says "fee"
        ↓
fee_rag_node: searches fee_structure.pdf → gets 4 relevant chunks
        ↓
response_node: LLM sees chunks + "you're a BCA student" → generates answer
        ↓
END → you print the answer

That's the entire system. Each node does exactly one job, the state is the shared context, and the graph defines the order. That's all LangGraph is at its core.
  """