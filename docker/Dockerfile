FROM continuumio/miniconda3

RUN conda update conda

RUN apt-get update && apt-get install libgl1-mesa-glx -y

COPY enviroment.yml .

RUN conda env create -f enviroment.yml

WORKDIR /conrec/src

CMD ["conda", "run", "--no-capture-output", "-n", "mainenv", "python3", "example.py"]
