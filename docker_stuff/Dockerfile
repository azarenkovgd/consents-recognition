FROM continuumio/miniconda3 AS build

COPY environment.yml .
RUN conda env create -f environment.yml

RUN conda install -c conda-forge conda-pack

RUN conda-pack -n example -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

RUN /venv/bin/conda-unpack


FROM debian:buster AS runtime

COPY --from=build /venv /venv


RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y


SHELL ["/bin/bash", "-c"]
ENTRYPOINT source /venv/bin/activate && \
           cd main && \
           python main.py
