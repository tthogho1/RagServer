import configparser
from flask import Flask, render_template
from flask import request
from ragClasses.OpenAIRag import OpenAIRag

config = configparser.ConfigParser()
config.read('config.ini')
db_name = config['vectordb']['name']

app = Flask(__name__, static_folder='static')
ragClass = OpenAIRag(db_name)


@app.route("/", methods=["GET"])
def hello_world():
    # if post method ,get parameter
    # static_file('index.html')
    return render_template('index.html')


@app.route("/load", methods=["GET", "POST"])
def loadWebSite():
    if request.method == 'POST':
        url = request.form.get('url')
        ragClass.setUrl(url)
        ragClass.loadUrl()
        return "load complete"
    else:
        return render_template('load.html')


@app.route("/answer", methods=["POST"])
def answerAboutSite():
    query = request.form.get('query')
    return ragClass.rag_chain(query)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8888, threaded=True)