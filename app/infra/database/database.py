"""/app/infra/database/database.py"""

from core.config import get_setting
from core.error.service_exception import ErrorCode, ServiceException
from core.log.logging import get_logging
from infra.database.base import Base
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

logger = get_logging()
settings = get_setting()


def get_database_url():
    """DB_ENGINE 설정에 따라 적절한 DATABASE_URL 생성"""
    if settings.DB_ENGINE == "postgresql":
        driver = "postgresql+psycopg2"
    elif settings.DB_ENGINE == "mysql":
        driver = "mysql+mysqlconnector"
    else:
        driver = settings.DB_ENGINE

    return (
        f"{driver}://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )


DATABASE_URL = get_database_url()


def get_engine() -> sessionmaker[Session]:
    logger.info(f"USE DATABASE {settings.DB_NAME}")

    engine = create_engine(
        DATABASE_URL,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_POOL_MAX_OVERFLOW,
        pool_recycle=settings.DB_POOL_RECYCLE,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
    )

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

    try:
        logger.info(f"Ensured all tables exist in {settings.DB_NAME}")
        Base.metadata.create_all(bind=engine)
    except ProgrammingError as e:
        logger.error(f"Error creating tables for {settings.DB_NAME}: {e}")
        raise

    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI 의존성으로 사용하는 단일 DB 세션 제공자."""
    SessionLocal = get_engine()
    session = SessionLocal()
    try:
        yield session
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        session.rollback()
        raise ServiceException(ErrorCode.DATABASE_ERROR)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
