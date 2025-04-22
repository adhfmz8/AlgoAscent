from fastapi import FastAPI
from pydantic import BaseModel
import db
from models import Problem, UserAttempt
from fastapi.middleware.cors import CORSMiddleware
# from recommender import recommend_ordered_problem, update_sm2, estimate_quality
from recommender_v5 import recommend_ordered_problem, process_attempt
from datetime import datetime

db.create_db_and_tables()
db.seed_database()

app = FastAPI()

origins = [
    "http://localhost:8000",
    "http://localhost",
    "null",
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
    return recommend_ordered_problem()


@app.post("/report")
async def report(data: Report):
    print("Report received:", data)

    # 1. Save the raw attempt
    attempt = UserAttempt(**data.model_dump())
    db.create_attempt(attempt)

    # 2. Update or create ProblemMemory using SM-2
    memory = db.get_or_create_problem_memory(data.problem_id)

    # 3. Update the ProblemMemory with the new attempt
    process_attempt(data.problem_id, data.solved, data.time_taken)

    return {"message": "Report processed using SM-2"}
