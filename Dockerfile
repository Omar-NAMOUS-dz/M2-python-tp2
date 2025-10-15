FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN pip install --upgrade ipython ipykernel

RUN ipython kernel install --name "tp2-env" --user

COPY . .

USER root

RUN chmod +x /app/script.sh

EXPOSE 5000

CMD ["/app/script.sh"]