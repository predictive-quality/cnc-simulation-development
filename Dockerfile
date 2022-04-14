
FROM tensorflow/tensorflow:latest-gpu as base

ADD ./ /code/
WORKDIR /code
RUN find . -name 'requirements.txt' -print  -exec pip install -r {} \;

ENTRYPOINT [ "python", "main.py"]
