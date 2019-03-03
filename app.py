# -*- coding: utf-8 -*-
import re
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_bootstrap import Bootstrap
import configparser
from src.correction import Checker

mode = 'docker'  # ('local', 'nlp', 'docker')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
bootstrap = Bootstrap(app)
URL_PREFIX = os.environ.get('URL_PREFIX', '')
if URL_PREFIX:
    URL_PREFIX = 'http://' + URL_PREFIX
ini = configparser.ConfigParser()
ini.read('./config.ini', 'UTF-8')
mecab_dict_file = ini.get(mode, 'mecab_dict_file')
model_file = ini.get(mode, 'model_file')
vocab_file = ini.get(mode, 'vocab_file')
opts_file = ini.get(mode, 'opts_file')
checker = Checker(mecab_dict_file, model_file, vocab_file, opts_file)


@app.route('/', methods=['GET', 'POST'])
def top():
    return render_template('checker.html')


@app.route('/api/correction', methods=['GET'])
def api():
    text = request.args.get('input_text')
    texts = re.split('[。．\t\n]', text)
    tokens = []
    for text in texts:
        text = text.strip()
        tokens += checker.correction_for_api(text)
        tokens += [["", 0]]
    return jsonify(({'tokens': tokens}))


if __name__ == '__main__':
    app.debug = False
    app.run(host='0.0.0.0', port=5001)
    # app.run(host='localhost', port=5001)
