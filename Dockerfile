#FROM ubuntu:20.04
FROM jjanzic/docker-python3-opencv

RUN apt-get update && apt-get install -y python3 python3-pip sudo

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

COPY ./src /app/src

COPY ./model /app/model

COPY ./static /app/static

COPY ./templates /app/templates

COPY ./app.py /app/app.py

COPY ./Dockerfile /app/Dockerfile

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]