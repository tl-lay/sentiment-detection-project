from flask import Flask,render_template,url_for,request,redirect, make_response, Response
import json
import pandas as pd
import numpy as np
from camera import VideoCamera

# url_for("index") ---> /
# url_for("video_prediction_screen") ---> /video-prediction-screen

app = Flask(__name__)  # initialize flask


# render home page
@app.route('/')  # home/index page
def index():
    return render_template('index.html')


@app.route('/video-prediction-screen')
def video_prediction_screen():
    return render_template('video-prediction-screen.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame
               + b'\r\n\r\n')


@app.route('/video-feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/data', methods=["GET", "POST"])
def data():
    data = pd.read_csv('predictions.csv').value_counts()
    neg = data.loc['negative'].values[0]
    neut = data.loc['neutral'].values[0]
    pos = data.loc['positive'].values[0]
    data = [neg, neut, pos]
    data = np.round(data/sum(data)*100, 2)
    data = data.tolist()
    response = make_response(json.dumps(data))

    response.content_type = 'application/json'

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)  # local host
