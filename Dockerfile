FROM python:3.10.6-slim AS base

COPY . /faceplugin-recognition
WORKDIR /faceplugin-recognition

RUN cp ./lib/* /usr/local/lib/
RUN ldconfig

RUN apt-get update && apt-get install -y libgomp1


RUN pip3 install -r requirements.txt

EXPOSE 8888

ENTRYPOINT ["python3"]
CMD ["app.py"]
