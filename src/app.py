from flask import Flask, request
from summarizer.summarizer import TextSummarizer

model = TextSummarizer()

app = Flask(__name__)

@app.route('/')
def welcome():
    return 'Use /summarize as POST'

@app.route('/summarize', methods=['POST'])
def summarize():
    json = request.json
    summarized_text = model.summarize(json['text'])
    # print(summarized_text)
    return {'summarization': summarized_text}
