import asyncpg

from credit_scoring.config import settings


async def check_database_connection() -> tuple[bool, str | None]:
    """Пинг базы данных PostgreSQL с помощью asyncpg."""
    try:
        url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(url, timeout=2.0)
        await conn.close()
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)
