import os

import cv2
import dlib
import face_recognition
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


sess, age, gender, train_mode, images_pl = load_network("./models")

video_capture = cv2.VideoCapture(0)

face_directory = "faces"

filenames = os.listdir(face_directory)

known_face_names = [filename.split(".")[0].split("_")[1] for filename in filenames]

known_face_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(os.path.join(os.getcwd(), face_directory, filename)))[0] for filename
                        in filenames]

img_size = 160

match_thresh = 0.49

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    img = frame

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_frame = frame[:, :, ::-1]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=160)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_h, img_w, _ = np.shape(rgb_frame)

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # detect faces using dlib detector
    detected = detector(rgb_frame, 1)
    faces = np.empty((1, img_size, img_size, 3))

    #     print("detected : ",detected)
    #     print("face_locations : ",face_locations)

    # Loop through each face in this frame of video
    for d, face_encoding in zip(detected, face_encodings):

        (top, right, bottom, left) = d.top(), d.right(), d.bottom(), d.left()

        faces[0, :, :, :] = fa.align(rgb_frame, gray, d)
        ages, genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})

        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        xmatches = face_recognition.face_distance(known_face_encodings, face_encoding)

        # print("xmatches : ",xmatches)
        # print("matches : ",matches)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            # name = known_face_names[first_match_index]
            idxx = np.argmin(xmatches)
            if xmatches[idxx] < match_thresh:
                name = known_face_names[idxx]

        label = "{}, {}, {}".format(name, int(ages[0]), "F" if genders[0] == 0 else "M")

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
