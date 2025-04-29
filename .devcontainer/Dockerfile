FROM quay.io/jupyter/minimal-notebook

USER jovyan

RUN mkdir -p /home/jovyan/temp

WORKDIR /home/jovyan/

COPY ../app.ipynb /home/jovyan/

COPY .devcontainer/requirements.txt .

RUN pip install --cache-dir=/home/jovyan/temp -r requirements.txt

EXPOSE 8888
EXPOSE 5000