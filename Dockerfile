From ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    git \
    pip 

RUN rm -rf /usr/lib/python3.12/EXTERNALLY-MANAGED

RUN git clone https://github.com/tthogho1/RagServer.git

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

EXPOSE 8000

WORKDIR RagServer
COPY config.ini .

CMD ["gunicorn", "--bind" , "0.0.0.0:8000" ,"-w", "2", "ragApp:app"]