import configparser
from flask import Flask
from flask import request
from ragClasses.OpenAIRag import OpenAIRag

config = configparser.ConfigParser()
# read config file
# openapikey="xxxxxx"
app = Flask(__name__)
ragClass = OpenAIRag()

@app.route("/", methods=["GET"])
def hello_world():
    # if post method ,get parameter     
    return app.send_static_file('index.html')

@app.route("/loading", methods=["GET"])
def loadingImage():
    # if post method ,get parameter     
    return app.send_static_file('img/loading.gif')

@app.route("/load", methods=["POST"])
def loadWebSite():
    url = request.form.get('url')
    ragClass.setUrl(url)
    ragClass.loadUrl()
    return "load complete"

@app.route("/answer", methods=["POST"])
def answerAboutSite():
    query= request.form.get('query')        
    message = ragClass.rag_chain(query)
    return message


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8888, threaded=True)