FROM ubuntu
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt update && apt install -y python3 python3-pip python3.12-venv

COPY requirements.txt /tmp/requirements.txt
# mmm, well upsie an hackathon
RUN pip3 install --break-system-packages -r /tmp/requirements.txt


WORKDIR /home/ubuntu

COPY src/* .

ENTRYPOINT ["fastapi", "run", "api.py"]