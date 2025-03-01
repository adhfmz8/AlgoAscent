# /Users/nathandiamond/Documents/GitHub/AlgoAscent/backend/models.py
from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime


class Problem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    leetcode_id: int = Field(unique=True)
    title: str
    url: str
    difficulty: str
    neetcode_order: Optional[int] = None
    neetcode_category: Optional[str] = None


class UserAttempt(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    problem_id: int = Field(foreign_key="problem.id")
    attempt_date: Optional[datetime] = Field(default=datetime.now())
    time_taken_minutes: Optional[float] = None
    solved: bool
    notes: Optional[str] = None
    last_attempt_date: Optional[datetime] = Field(default=datetime.now())
    easiness_factor: Optional[float] = Field(
        default=2.5
    )  # Default value of easiness factor
    next_review_date: Optional[datetime] = Field(
        default=datetime.now()
    )  # Default next review date
