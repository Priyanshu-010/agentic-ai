# So now we are creating a graph 
# and the first thing you are creating is a state

import os

# 1) Typed Dictionary (Most Common approach)
# It is good at data validationa and type checking at compile time

from typing import TypedDict

class State(TypedDict):
  topic: str
  summary: str
  score: int

# 2) Pydantic Approach
# It is good at data validationa and type checking at run time

from pydantic import BaseModel, field_validator

class State(BaseModel):
  topic: str
  summary: str = ""
  score: int

  @field_validator("score")
  def score_positive(cls,v):
    if v<0:
      raise ValueError("score must be positive")
    return v

# 3) Dataclass Approach
# It is good at data validationa and type checking at run time
# Standard Python Dataclass but it is used very rarely

from dataclasses import dataclass, field

@dataclass
class State:
  topic: str = ""
  summary: str = ""
  score: int = field(default=0)
  massages: list[str] = field(default_factory=list)

from langgraph.graph import MessagesState

class State(MessagesState):
  # messages field is already included with add_message reducer
  # just add your extra fields

  user_name: str
  language: str