FROM python:3.7
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN rm requirements.txt
WORKDIR /opt/
RUN curl -O http://fastjet.fr/repo/fastjet-3.3.3.tar.gz
RUN tar zxvf fastjet-3.3.3.tar.gz
WORKDIR /opt/fastjet-3.3.3/
RUN ./configure --enable-pyext --prefix=$PWD/../fastjet-install
RUN make
RUN make check
RUN make install
ENV LD_LIBRARY_PATH=/opt/fastjet-install/lib
ENV PYTHONPATH=/opt/fastjet-install/lib/python3.7/site-packages
WORKDIR /
