from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

SQLALCHEMY_DATABASE_URL = "sqlite:///./voxtube.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def run_migrations(engine):
    """Add new columns that didn't exist in older DB versions."""
    from sqlalchemy import text
    migrations = [
        "ALTER TABLE comments ADD COLUMN lang VARCHAR",
        "ALTER TABLE comments ADD COLUMN published_at DATETIME",
        "ALTER TABLE jobs ADD COLUMN view_count INTEGER",
        "ALTER TABLE jobs ADD COLUMN like_count INTEGER",
        "ALTER TABLE jobs ADD COLUMN channel_title VARCHAR",
    ]
    with engine.connect() as conn:
        for stmt in migrations:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:
                pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
