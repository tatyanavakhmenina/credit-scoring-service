# Credit Scoring MLOps Service

Асинхронный микросервис на базе **FastAPI** для инференса модели кредитного скоринга, разработанный в рамках учебного курса по MLOps. Проект реализует принципы production-ready разработки: строгий статический анализ, покрытие тестами, контейнеризацию и полный CI/CD пайплайн.

---

## Технологии и стек

* **Язык:** Python 3.11+
* **Веб-фреймворк:** FastAPI (асинхронное приложение)
* **Пакетный менеджер:** `uv`
* **База данных:** PostgreSQL (`asyncpg`)
* **Валидация и конфиги:** Pydantic / Pydantic-Settings
* **Линтинг и форматирование:** Ruff
* **Тестирование:** Pytest, Pytest-Asyncio, HTTPX, Pytest-Cov
* **Контейнеризация:** Docker (Multi-stage builds, non-root user), Docker Compose
* **CI/CD:** GitHub Actions, GitHub Container Registry (GHCR)

---

## Структура проекта

```text
credit-scoring-service/
├── .github/
│   └── workflows/
│       ├── ci.yml          # CI пайплайн (линтеры, тесты, покрытие)
│       └── cd.yml          # CD пайплайн (сборка и пуш Docker-образа)
├── src/
│   └── credit_scoring/
│       ├── __init__.py
│       ├── main.py         # Точка входа FastAPI
│       ├── config.py       # Конфигурация через pydantic-settings
│       ├── database.py     # Асинхронное подключение и пинг БД
│       └── api/
│           └── v1/
│               ├── endpoints.py # Эндпоинты API
│               └── router.py
├── tests/
│   └── test_app.py         # Асинхронные тесты (pytest + httpx)
├── .pre-commit-config.yaml # Конфигурация пре-комит хуков
├── pyproject.toml          # Зависимости и настройки проекта
├── Dockerfile              # Multi-stage Docker образ
└── docker-compose.yml      # Оркестрация приложения и PostgreSQL
