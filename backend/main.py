# /Users/nathandiamond/Documents/GitHub/AlgoAscent/backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import db
from models import Problem, UserAttempt
from fastapi.middleware.cors import CORSMiddleware
from recommender import (
    recommend_ordered_problem,
    calculate_new_easiness_factor,
    calculate_next_review_date,
)
from datetime import datetime, timedelta

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
    attempt = UserAttempt(**data.model_dump())

    # Fetch the last attempt to update easiness
    last_attempt = db.get_last_attempt(attempt.problem_id)
    if last_attempt:
        attempt.easiness_factor = calculate_new_easiness_factor(
            last_attempt, attempt.solved, attempt.time_taken
        )
    else:
        attempt.easiness_factor = 2.5  # default easiness factor

    attempt.next_review_date = calculate_next_review_date(attempt.easiness_factor)

    db.create_attempt(attempt)
    return {"message": "Report received"}
