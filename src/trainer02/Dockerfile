FROM python:3.7

WORKDIR /usr/src/app

COPY . .

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | /usr/local/bin/python -
ENV PATH $PATH:/root/.poetry/bin
RUN poetry config virtualenvs.in-project true
RUN poetry install

ENTRYPOINT [ "poetry", "run", "python" , "train.py"]