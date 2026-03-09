import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Variables de entorno - Configuración PostgreSQL
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "password")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")

# Si se proporciona INSTANCE_CONNECTION_NAME, asumimos Cloud Run
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")

def get_engine():
    if INSTANCE_CONNECTION_NAME:
        # Conexión vía Unix Socket para Google Cloud Run + Cloud SQL PostgreSQL
        url = (
            f"postgresql+pg8000://{DB_USER}:{DB_PASS}@/{DB_NAME}"
            f"?unix_sock=/cloudsql/{INSTANCE_CONNECTION_NAME}/.s.PGSQL.5432"
        )
    else:
        # Desarrollo local TCP
        url = f"postgresql+pg8000://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    return create_engine(
        url,
        pool_pre_ping=True,         # evita conexiones muertas
        pool_recycle=1800,          # recicla cada 30 min
        pool_size=5, max_overflow=5 # límites sensatos
    )