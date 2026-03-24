FROM sphinxdoc/sphinx:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --quiet --yes --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /ugtsdti/requirements.txt
COPY docs/requirements.txt /ugtsdti/docs/requirements.txt
RUN pip3 install -r /ugtsdti/requirements.txt -r /ugtsdti/docs/requirements.txt && \
    rm -rf /root/.cache

COPY . /ugtsdti
WORKDIR /ugtsdti

EXPOSE 8000
ENTRYPOINT sphinx-autobuild --host 0.0.0.0 --port 8000 --watch . docs/source /tmp/ugtsdti/docs
