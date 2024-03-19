FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y pip
RUN apt-get install -y python3.10-venv

WORKDIR /app
COPY requirements.txt .
COPY app.py .

COPY modelx_trace.nc .

RUN rm -rf venv
RUN python3 -m venv venv

RUN venv/bin/pip install -r requirements.txt


CMD . venv/bin/activate && exec python3 app.py 

EXPOSE 8080
