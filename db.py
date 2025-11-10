from sqlalchemy import create_engine

# --- TU MYSQL EN RAILWAY (host interno) ---
MYSQL_HOST = "mysql.railway.internal"
MYSQL_PORT = 3306
MYSQL_DB   = "railway"
MYSQL_USER = "root"
MYSQL_PASS = "EDXzpjNKykmenwtSNlnlXnpGmckkIgRx"

def get_engine():
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
    )
    return create_engine(
        url,
        pool_pre_ping=True,         # evita conexiones muertas
        pool_recycle=1800,          # recicla cada 30 min
        pool_size=5, max_overflow=5 # límites sensatos
    )