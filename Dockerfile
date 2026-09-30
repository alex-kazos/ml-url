# Lean image for the inference API: same dependencies as the Render deploy.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY Classes/ Classes/
COPY Services/ Services/
COPY Utilities/ Utilities/
COPY Models/ Models/

EXPOSE 8000
CMD ["uvicorn", "Services.api:app", "--host", "0.0.0.0", "--port", "8000"]
