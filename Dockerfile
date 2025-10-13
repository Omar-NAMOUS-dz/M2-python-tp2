FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD "jupyter nbconvert --to html --execute data_analysis.ipynb --output data_analysis.html --output-dir ./templates/ && python app.py"