import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

def get_engine():
    user = os.getenv("MYSQL_USER", "root")
    pwd  = os.getenv("MYSQL_PASS", "")
    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = os.getenv("MYSQL_PORT", "3306")
    db   = os.getenv("MYSQL_DB", "sw2_examen")
    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)
