FROM fedora
RUN dnf install -y python3.10 git curl

# Break system packages? Hackathon shortcut here
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3.10 /tmp/get-pip.py --break-system-packages

COPY models /mnt/models
COPY requirements.txt /tmp/requirements.txt


# mmm, well upsie an hackathon
RUN pip3.10 install --break-system-packages -r /tmp/requirements.txt


WORKDIR /home/ubuntu

COPY src/* .

ENTRYPOINT ["fastapi", "run", "api.py"]