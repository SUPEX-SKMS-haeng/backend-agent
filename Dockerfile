FROM ghcr.io/astral-sh/uv:0.9.13 AS uv

FROM python:3.12-slim

# Install uv
COPY --from=uv /uv /usr/local/bin/uv

RUN apt update && \
    apt install -y binutils && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY ./app/ ./

CMD ["uv", "run", "python3", "main.py"]
