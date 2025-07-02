import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from .env
load_dotenv()

# Database connection string
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('POSTGRES_HOST')
DB_PORT = os.getenv('POSTGRES_PORT')
DB_NAME = os.getenv('POSTGRES_DB')
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Future: Add shared utility functions, tool registry, and data models here 