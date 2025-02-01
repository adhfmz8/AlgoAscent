# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import db
from models import Problem, UserAttempt
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from recommender import recommend_ordered_problem

db.create_db_and_tables()
db.seed_database()

app = FastAPI()

origins = [
    "http://localhost:8000",  # the origin where the server is running
    "http://localhost",  # localhost in general
    "null",  # electron sends the origin as null by default
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Report(BaseModel):
    problem_id: int
    time_taken: float  # in minutes
    solved: bool


@app.get("/recommend", response_model=Problem)
async def recommend():
    # A placeholder recommendation algorithm: pick a random problem
    problem = recommend_ordered_problem()
    return problem


@app.post("/report")
async def report(data: Report):
    # Here you would process the data (e.g., update user stats, adjust recommendation parameters)
    # For now, just log the data and return a simple acknowledgement.
    print("Report received:", data)
    db.create_attempt(UserAttempt(**data.model_dump()))
    return {"message": "Report received"}
