FROM python:3.8

RUN mkdir /usr/src/app
COPY . /usr/src/app
WORKDIR /usr/src/app
EXPOSE 5000

RUN pip install --upgrade pip
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install transformers nltk networkx numpy Flask
RUN python -c "import nltk; nltk.download('popular')"

ENV FLASK_APP "app.py"
CMD flask run --host=0.0.0.0