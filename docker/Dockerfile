FROM ubuntu:latest

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils git 

RUN git clone https://github.com/HexaFarms/GraphCut.git /GraphCut
WORKDIR /GraphCut
RUN pip install -r requirements.txt

RUN git clone https://github.com/Borda/pyGCO.git  pyGCO
WORKDIR "./pyGCO"
RUN pip install -r requirements.txt
RUN python setup.py install
WORKDIR "/GraphCut"
