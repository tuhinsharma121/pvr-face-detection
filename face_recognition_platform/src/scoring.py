import face_recognition
import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
from imutils.face_utils import FaceAligner
from face_recognition_platform.src import inception_resnet_v1




def load_network(model_path):
    sess = tf.Session()
    images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
    images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
    train_mode = tf.placeholder(tf.bool)
    age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                 phase_train=train_mode,
                                                                 weight_decay=1e-5)
    gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
    age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
    age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restore model!")
    else:
        pass
    return sess, age, gender, train_mode, images_pl




def load_parameters():

    img_size = 160
    match_thresh = 0.49
    face_directory = "faces"
    sess, age, gender, train_mode,images_pl = load_network("./models")


    filenames = os.listdir(face_directory)
    known_ids = [filename.split(".")[0].split("_")[0] for filename in filenames]
    known_face_names = [filename.split(".")[0].split("_")[1] for filename in filenames]
    known_face_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(os.path.join(os.getcwd(),face_directory,filename)))[0] for filename in filenames]


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=160)

    return img_size,match_thresh,sess, age, gender, train_mode,images_pl,known_face_names,known_face_encodings,detector,fa,known_ids


def recognize(frame,img_size,match_thresh,sess, age, gender, train_mode,images_pl,known_face_names,known_face_encodings,detector,fa,known_ids):
    img = frame

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_h, img_w, _ = np.shape(rgb_frame)

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # detect faces using dlib detector
    detected = detector(rgb_frame, 1)
    faces = np.empty((1, img_size, img_size, 3))

    result_list = []

    for d, face_encoding in zip(detected, face_encodings):

        result = dict()

        (top, right, bottom, left) = d.top(), d.right(), d.bottom(), d.left()

        faces[0, :, :, :] = fa.align(rgb_frame, gray, d)
        ages, genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        xmatches = face_recognition.face_distance(known_face_encodings, face_encoding)

        # print("xmatches : ",xmatches)
        # print("matches : ",matches)

        name = "Unknown"
        id = -1

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            idx = np.argmin(xmatches)
            if xmatches[idx] < match_thresh:
                name = known_face_names[idx]
                id=known_ids[idx]

        person_age = int(ages[0])
        person_gender = "F" if genders[0] == 0 else "M"
        person_bounding_box = (top, right, bottom, left)
        person_bounding_box_area = (bottom - top) * (right - left)

        result['name'] = name
        result['id'] = id
        result['age'] = person_age
        result['gender'] = person_gender
        result['bounding_box'] = person_bounding_box
        result['bounding_box_area'] = person_bounding_box_area

        result_list.append(result)

    return result_list

if __name__ == "__main__":

    img_size,match_thresh,sess, age, gender, train_mode,images_pl,known_face_names,known_face_encodings,detector,fa,known_ids = load_parameters()
    frame = cv2.imread("data/sample.jpg")
    res = recognize(frame,img_size,match_thresh,sess, age, gender, train_mode,images_pl,known_face_names,known_face_encodings,detector,fa,known_ids)
    print(res)