FROM tensorflow/tensorflow:1.13.1-gpu-py3

#RUN apt-get update && apt-get -y upgrade

RUN pip3 install keras
RUN pip3 install tqdm
RUN pip3 install pydot
RUN pip3 install sklearn
RUN pip3 install scipy
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install psutil
RUN pip3 install Pillow
#RUN pip3 install guppy==0.1.10
RUN pip3 install hyperopt
RUN	pip3 install grpcio==1.22.0
RUN pip3 install matplotlib
#RUN pip install pickle
#RUN pip install bayesian-optimization