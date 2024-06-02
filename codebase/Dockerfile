FROM python:3.10

WORKDIR /opt

COPY . .

RUN apt-get update -qq && apt-get -y install libgl1

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]
