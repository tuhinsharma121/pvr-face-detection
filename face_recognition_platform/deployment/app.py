import flask
import logging
from flask import Flask, request
import sys
from werkzeug.utils import secure_filename
import os
import io
import cv2
import base64
import numpy as np
from PIL import Image
from flask_cors import CORS

from face_recognition_platform.src.scoring import load_parameters,recognize
from face_recognition_platform.src.movie_rec import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def purchased_before(user_id):
    user = str(user_id)
    df1 = pd.read_csv("models/Movie-Data.csv",index_col=0)
    df1['User_ID'] = df1.User_ID.astype(object)
    df2=df1.loc[df1['User_ID'] == user, df1.columns[2:6]]
    df2 = df1[df1.User_ID==int(user)][df1.columns[2:6]]
    for elem in range(4):
        if df2.iloc[0][elem] != 0:
            return True
    return False

global imdb_recsys
try:
    app.img_size, app.match_thresh, app.sess, app.age, app.gender, app.train_mode, app.images_pl, app.known_face_names, \
    app.known_face_encodings, app.detector, app.fa, app.known_ids= load_parameters()
except:
    raise("failed to initialize model parameters!!")


def base64_to_cv2(base64_string):
    imgdata = base64.b64decode(base64_string)
    x = Image.open(io.BytesIO(imgdata))
    y = cv2.cvtColor(np.array(x), cv2.COLOR_BGR2RGB)
    return y

@app.route('/')
def heart_beat():
    return flask.jsonify({"status": "ok"})


@app.route('/predict', methods=['POST'])
def score():
    request_json = request.get_json()

    base64_image = request_json['image']

    frame = base64_to_cv2(base64_image)
    res = recognize(frame, app.img_size, app.match_thresh, app.sess, app.age, app.gender, app.train_mode, app.images_pl, app.known_face_names,
                    app.known_face_encodings, app.detector, app.fa,app.known_ids)
    return flask.jsonify(res)


@app.route('/food/<id>',methods=['GET'])
def food_recommendations(id):
    user_id = int(id)
    if user_id is -1:
        return food_cold_no_login()
    else:
        if purchased_before(user_id) is True:
            return food_personalised(user_id)
        else:
            return food_cold_first_login(user_id)

@app.route('/movies/<id>',methods=['GET'])
def movie_recommendations(id):
    user_id = int(id)
    if user_id is -1:
        return movie_cold_no_login()
    else:
        if purchased_before(user_id) is True:
            return movie_personalised(user_id)
        else:
            return movie_first_login(user_id)



if __name__ == "__main__":
    app.run()
