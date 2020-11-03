from flask import Flask, jsonify, request
from modules.data_extractor import DataExtractor
import os

from config import api

app = Flask(__name__)

@app.route('/', methods=['GET'])
def about():
    jsn = {'app': 'Test'}
    return jsonify(jsn)


@app.route('/data_extractor/', methods=['POST'])
def get_value():
    data_extr = DataExtractor(request.json)
    data_extr.check_json()
    if len(data_extr.error):
        return jsonify(data_extr.error)
    data_extr.create_df()
    if "split_text" in data_extr.json:
        data_extr.split_text()
    if "classify_text" in data_extr.json:
        data_extr.classify_text(api_params=api)
    data_extr.regex_val()
    data_extr.get_res_json()

    return jsonify(data_extr.res_json)


if __name__ == '__main__':
    this_host = os.environ['this_host']
    this_port = os.environ['this_port']
    app.run(host='0.0.0.0', port=this_port, debug=False)
