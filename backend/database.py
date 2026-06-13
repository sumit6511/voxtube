from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

SQLALCHEMY_DATABASE_URL = "sqlite:///./voxtube.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}  # required for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def run_migrations(engine):
    """
    Add new columns that didn't exist in older DB versions.
    Safe to run on every startup — silently skips columns that already exist.
    """
    from sqlalchemy import text
    migrations = [
        "ALTER TABLE comments ADD COLUMN lang VARCHAR",
        "ALTER TABLE comments ADD COLUMN published_at DATETIME",
        "ALTER TABLE comments ADD COLUMN parent_id VARCHAR",
    ]
    with engine.connect() as conn:
        for stmt in migrations:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:
                pass   # column already exists — ignore


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
