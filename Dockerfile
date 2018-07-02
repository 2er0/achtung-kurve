FROM python:3.6
MAINTAINER David Baumgartner

# Some stuff that everyone has been copy-pasting
# since the dawn of time.
ENV PYTHONUNBUFFERED 1
ENV AGENTS 2

# Install some necessary things.
RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y swig libssl-dev dpkg-dev netcat sqlite3 python3-pip

# Copy all our files into the image.
RUN mkdir -p /kurvSrv/achtungkurve
WORKDIR /kurvSrv
COPY setup.py LICENSE /kurvSrv/
COPY achtungkurve /kurvSrv/achtungkurve/

# Install our requirements.
RUN pip install -U pip && python setup.py install

EXPOSE 15555

CMD ["python", "/kurvSrv/achtungkurve/server.py"]