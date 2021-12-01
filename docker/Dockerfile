RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/HexaFarms/GraphCut.git /GraphCut
WORKDIR /GraphCut
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt

RUN git clone https://github.com/Borda/pyGCO.git  /GraphCut/pyGCO
RUN pip install -r pyGCO/requirements.txt
RUN python pyGCO/setup.py
