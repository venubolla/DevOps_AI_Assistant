from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

# creates the connection infrastructure.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)

# creates database sessions.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


# will later allow FastAPI endpoints to access PostgreSQL safely.
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
