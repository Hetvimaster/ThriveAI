import psycopg
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()
DB_URI = os.getenv("DATABASE_URL")
if not DB_URI:
    raise ValueError("DATABASE_URL environment variable is missing")

conn = psycopg.AsyncConnection.connect(DB_URI)

print("Connected!")