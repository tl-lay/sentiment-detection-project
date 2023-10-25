from flask import Flask,render_template,url_for,request,redirect, make_response, Response
import json
import pandas as pd
import numpy as np
from camera import VideoCamera
import os
import csv

# url_for("index") ---> /
# url_for("video_prediction_screen") ---> /video-prediction-screen

app = Flask(__name__)  # initialize flask


@app.route('/')  # home/index page
def index():
    with open('predictions.csv', 'w') as f:
        writerObj = csv.writer(f)
        writerObj.writerow(['sentiment'])
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



# Endpoint to provide real-time emotion data for the line chart
@app.route('/data', methods=["GET", "POST"])
def data():
    if request.args.get('realtime') == '1':
        if os.path.exists('predictions.csv'):
            data = pd.read_csv('predictions.csv')
            unique_values = data['sentiment'].unique()
            data = data.value_counts()
            if 'negative' in unique_values:
                neg = data.loc['negative'].values[0]
            else:
                neg = 0
            if 'neutral' in unique_values:
                neut = data.loc['neutral'].values[0]
            else:
                neut = 0
            if 'positive' in unique_values:
                pos = data.loc['positive'].values[0]
            else:
                pos = 0
            data = np.array([neg, neut, pos])
        else:
            data = [0, 0, 0]

        if sum(data) > 0:
            data = np.round(data / sum(data) * 100, 2)
            data = data.tolist()
        else:
            data = data.tolist()

    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)


