FROM python:3.6
COPY simulators/ ./simulators
COPY configs/ ./configs 
COPY flow/ ./flow 
COPY influence/ ./influence 
COPY plots/ ./plots 
COPY scripts/ ./scripts
COPY experiment.py ./
COPY trainer.py ./
COPY logs.monitor.csv ./
RUN apt-get update -y
RUN apt-get install -y git cmake python3 g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev python3-dev swig default-jdk maven libeigen3-dev
RUN git clone --recursive https://github.com/eclipse/sumo
RUN export SUMO_HOME="$PWD/sumo"
CMD mkdir ./sumo/sumo/build/cmake-build
CMD cd sumo/build/cmake-build
CMD cmake ../..
CMD make -j$(nproc)
CMD cd /
RUN git clone https://github.com/INFLUENCEorg/flow.git
RUN git clone https://github.com/miguelsuau/recurrent_policies.git
RUN pip install -e ./flow
RUN pip install stable-baselines3
RUN pip install numpy
RUN pip install matplotlib
RUN pip install pyaml
RUN pip install sacred
RUN pip install pymongo
RUN pip install sshtunnel
RUN pip install networkx
RUN pip install lxml
RUN pip install pyglet
RUN pip install imutils
RUN pip install scipy
RUN pip install torch==1.8.1
RUN pip install pathos
RUN pip install psutil
RUN pip install opencv-python==4.5.3.56
RUN pip install -e ./simulators/warehouse/
RUN pip install -e simulators/traffic/
CMD chmod -R 777 /usr/local/lib/python3.6/site-packages
CMD chmod -R 777 /flow/
CMD export SUMO_HOME="/sumo"
CMD export PYTHONPATH="${SUMO_HOME}/tools:${PYTHONPATH}"
CMD export PATH="${SUMO_HOME}/bin:${PATH}"