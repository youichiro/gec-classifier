# -*- coding: utf-8 -*-
import re
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_bootstrap import Bootstrap
import configparser
from src.correction import Checker as CheckerV1
from src.correction_particle19 import Checker as CheckerV2

mode = 'local'  # ('local', 'nlp', 'docker')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
bootstrap = Bootstrap(app)
URL_PREFIX = os.environ.get('URL_PREFIX', '')
if URL_PREFIX:
    URL_PREFIX = '//' + URL_PREFIX

# ver.1
ini = configparser.ConfigParser()
ini.read('./config.ini', 'UTF-8')
mecab_dict_file = ini.get(mode, 'mecab_dict_file')
model_file = ini.get(mode, 'model_file')
vocab_file = ini.get(mode, 'vocab_file')
opts_file = ini.get(mode, 'opts_file')
checker_v1 = CheckerV1(mecab_dict_file, model_file, vocab_file, opts_file)

# ver.2
ini = configparser.ConfigParser()
ini.read('./config_v2.ini', 'UTF-8')
mecab_dict_file = ini.get(mode, 'mecab_dict_file')
model_file = ini.get(mode, 'model_file')
vocab_file = ini.get(mode, 'vocab_file')
opts_file = ini.get(mode, 'opts_file')
reverse = False
threshold = 0.7
checker_v2 = CheckerV2(mecab_dict_file, model_file, vocab_file, opts_file,
                       reverse=reverse, threshold=threshold)


@app.route('/', methods=['GET', 'POST'])
def top():
    return render_template('checker.html', prefix=URL_PREFIX)


@app.route('/api/correction', methods=['GET'])
def api():
    text = request.args.get('input_text')
    texts = re.split('[。．\t\n]', text)
    tokens = []
    for text in texts:
        text = text.strip()
        tokens += checker_v1.correction_for_api(text)
        tokens += [["", 0]]
    return jsonify(({'tokens': tokens}))


@app.route('/v2/api/correction', methods=['GET'])
def v2_correction():
    text = request.args.get('input_text')
    if not text:
        return ''
    text = text.strip()
    res = checker_v2.correction(text)
    return res


if __name__ == '__main__':
    app.debug = False
    if mode == 'local':
        app.run(host='localhost', port=5001)
    else:
        app.run(host='0.0.0.0', port=5001)
