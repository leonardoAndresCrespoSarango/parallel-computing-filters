FROM  nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get -qq update                  && \
    apt-get -qq install build-essential    \
    python3-pip python3-dev && \
    pip3 install pycuda




COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0

RUN pip3 install -r requirements.txt


COPY . .

EXPOSE 8080

# Define el comando para ejecutar la aplicación
CMD ["python3", "app.py"]


