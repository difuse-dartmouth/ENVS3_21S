FROM python:3.8-slim-buster

EXPOSE 8000

WORKDIR /env3

COPY requirements.txt requirements.txt
RUN pip install -v https://github.com/pallets/werkzeug/archive/refs/tags/2.0.3.tar.gz
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:server"]
