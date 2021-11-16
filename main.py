# pip install onnxruntime-gpu
# pip install opencv-python

import cv2
import numpy as np
import onnxruntime

import time
from datetime import datetime

import face_utils
import utils

import threading
import sys
import os
from PyQt5 import QtWidgets
from PyQt5 import QtGui
# from PyQt5 import QtCore

# Load 3DDFA(facial landmark model) parameter
bfm = face_utils._load('configs/bfm_slim.pkl')
u = bfm.get('u').astype(np.float32)  # fix bug
w_shp = bfm.get('w_shp').astype(np.float32)[..., :50]
w_exp = bfm.get('w_exp').astype(np.float32)[..., :12]
tri = bfm.get('tri')
tri = face_utils._to_ctype(tri.T).astype(np.int32)
keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
w = np.concatenate((w_shp, w_exp), axis=1)
w_norm = np.linalg.norm(w, axis=0)
u_base = u[keypoints].reshape(-1, 1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]

# params normalization config
r = face_utils._load('configs/param_mean_std_62d_120x120.pkl')
param_mean = r.get('mean')
param_std = r.get('std')

def img_convert(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (105, 105))
    image = np.expand_dims(image, axis=(0, 1))  # (105, 105) → (1, 1, 105, 105)
    image = image.astype(np.float32)
    return image

def crop_periocular(img, box):
    param = lm_session.run(None, {'input': img})[0]
    param = param.flatten().astype(np.float32)
    param = param * param_std + param_mean  # re-scale
    vers = face_utils.recon_vers([param], [[box[0], box[1], box[2], box[3]]], u_base, w_shp_base, w_exp_base)[0]

    # 랜드마크의 최소/최대 좌표 계산
    # 원본이미지 + 랜드마크 → vers[x, y][landmark index]
    x_max = np.max(vers[0][:])
    x_min = np.min(vers[0][:])
    y_min = np.min(vers[1][:])

    # face_point 저장
    face_point = np.zeros((2, 5))  # face_point[x, y][point_idx]
    face_point[0][0] = vers[0][36]  # point1 - right point of right eye
    face_point[1][0] = vers[1][36]
    face_point[0][1] = vers[0][39]  # point2 - left point of right eye
    face_point[1][1] = vers[1][39]
    face_point[0][2] = vers[0][42]  # point3 - right point of left eye
    face_point[1][2] = vers[1][42]
    face_point[0][3] = vers[0][45]  # point4 - left point of left eye
    face_point[1][3] = vers[1][45]
    face_point[0][4] = vers[0][30]  # point5 - point of nose
    face_point[1][4] = vers[1][30]

    # ROI 영역 계산
    # point 1, 2, 3, 4의 y좌표 평균(eye line) 계산
    eye_y = int(0)

    for i in range(0, 4):
        eye_y = eye_y + face_point[1][i]
    eye_y = eye_y / 4.0

    # 눈과 코 사이 수직 거리 (Vertical eye-to-nose distance) 계산
    dist = eye_y - abs(face_point[1][4] - eye_y)

    # 눈썹 부분 보존
    if y_min > dist:  # 부호를 반대로하면 눈썹 부분이 사라질 수 있음
        y_min = dist

    y_max = face_point[1][4]

    return int(x_min), int(x_max), int(y_min), int(y_max), vers

def run():
    global running, euclidean_distance
    global enrolling
    global matching
    global capture_time
    global pass_state
    preTime = 0
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    pixmap_label.resize(int(width), int(height))
    if matching:
        capture_time = time.time()
    while running:
        success, captured_image = cap.read()
        if success:
            captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
            # processing_time = int(time.time() - capture_time)
            # cv2.putText(captured_image, "Count : %d" % (9 - processing_time), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if enrolling:
                processing_time = int(time.time() - capture_time)
                cv2.putText(captured_image, "Count : %d" % (9 - processing_time), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if processing_time >= 10:
                    captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)  # image(480, 640, 3)

                    file_name = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    save_path = enrolled_dir + file_name + '.jpg'
                    cv2.imwrite(save_path, captured_image)
                    print("Enrolled!")
                    capture_time = time.time()

            elif matching:
                # cv2.putText(captured_image, "Count : %d" % (2 - processing_time), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB) # CV BGR -> RGB(모델에 맞게)
                image = cv2.resize(image, (320, 240))
                image_mean = np.array([127, 127, 127])
                image = (image - image_mean) / 128
                image = np.transpose(image, [2, 0, 1])
                image = np.expand_dims(image, axis=0)
                image = image.astype(np.float32)

                cap_confidences, cap_boxes = fd_session.run(None, {fd_input_name: image})
                cap_boxes, cap_labels, cap_probs = utils.predict(captured_image.shape[1], captured_image.shape[0], cap_confidences, cap_boxes, 0.9)

                for i in range(cap_boxes.shape[0]):
                    cap_box = cap_boxes[0, :]
                    cap_label = f"{fd_classes_name[cap_labels[i]]}: {cap_probs[i]:.2f}"
                    cap_box_margin = int((cap_box[2] - cap_box[0]) / 6)
                    # add box margin for accurate face landmark inference
                    cap_box = [cap_box[0] - cap_box_margin, cap_box[1] - cap_box_margin, cap_box[2] + cap_box_margin, cap_box[3] + cap_box_margin]
                    cropped_img = captured_image[cap_box[1]:cap_box[3], cap_box[0]:cap_box[2]].copy()
                    cropped_img = cv2.resize(cropped_img, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
                    cropped_img = cropped_img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
                    cropped_img = (cropped_img - 127.5) / 128.0
                    x_min, x_max, y_min, y_max, vers = crop_periocular(cropped_img, cap_box)

                    # Draw face landmark
                    for i in range(68):
                        cv2.circle(captured_image, (int(vers[0, i]), int(vers[1, i])), 1, (0, 255, 0), -1,
                                   cv2.LINE_AA)

                    if pass_state:
                        dist_str = 'Pass: %.2f' % np.min(euclidean_distance)
                        box_color = (0, 255, 255)
                    else:
                        dist_str = 'Fail: %.2f' % np.min(euclidean_distance)
                        box_color = (255, 0, 0)
                    cv2.rectangle(captured_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), box_color, 4)
                    cv2.putText(captured_image, dist_str, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, box_color, 2)
                    cv2.rectangle(captured_image, (cap_box[0], cap_box[1]), (cap_box[2], cap_box[3]), (0, 255, 0),
                                  4)
                    cv2.putText(captured_image, cap_label,
                                (cap_box[0], cap_box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,  # font scale
                                (255, 0, 255), 2)  # line color and type

                    # if processing_time >= 3:
                    # test image crop
                    test_cap_image = captured_image[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1]
                    test_cap_image = img_convert(test_cap_image)  # numpy(1, 1, 105, 105)

                    enrolled_img_list = os.listdir(enrolled_dir)
                    euclidean_distance = np.zeros(len(enrolled_img_list))

                    for idx in range(len(enrolled_img_list)):
                        enrolled_img_path = enrolled_dir + enrolled_img_list[idx]
                        # print(enrolled_img_path)
                        saved_img = cv2.imread(enrolled_img_path, cv2.COLOR_BGR2RGB) # image(480, 640, 3)
                        enrolled_img = cv2.cvtColor(saved_img, cv2.COLOR_BGR2RGB) # CV BGR -> RGB(모델에 맞게)
                        enrolled_img = cv2.resize(enrolled_img, (320, 240))
                        enrolled_image_mean = np.array([127, 127, 127])
                        enrolled_img = (enrolled_img - enrolled_image_mean) / 128
                        enrolled_img = np.transpose(enrolled_img, [2, 0, 1])
                        enrolled_img = np.expand_dims(enrolled_img, axis=0)
                        enrolled_img = enrolled_img.astype(np.float32)
                        db_confidences, db_boxes = fd_session.run(None, {fd_input_name: enrolled_img})
                        db_boxes, db_labels, db_probs = utils.predict(saved_img.shape[1], saved_img.shape[0], db_confidences, db_boxes, 0.9)

                        for i in range(db_boxes.shape[0]):
                            db_box = db_boxes[0, :]
                            # db_label = f"{fd_classes_name[db_labels[i]]}: {db_probs[i]:.2f}"
                            db_box_margin = int((db_box[2] - db_box[0]) / 6)
                            # add box margin for accurate face landmark inference
                            db_box = [db_box[0] - db_box_margin, db_box[1] - db_box_margin, db_box[2] + db_box_margin, db_box[3] + db_box_margin]
                            cropped_db_img = saved_img[db_box[1]:db_box[3], db_box[0]:db_box[2]].copy()
                            cropped_db_img = cv2.resize(cropped_db_img, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
                            cropped_db_img = cropped_db_img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
                            cropped_db_img = (cropped_db_img - 127.5) / 128.0
                            x_min, x_max, y_min, y_max, vers = crop_periocular(cropped_db_img, db_box)

                        test_db_image = saved_img[int(y_min):int(y_max) + 1, int(x_min):int(x_max) + 1]
                        test_db_image = img_convert(test_db_image)  # crop(64, 91, 3) → numpy(1, 1, 105, 105)

                        # vec_session = onnxruntime.InferenceSession('./onnx/' + target + '.onnx', None)
                        vec1, vec2 = vec_session.run([vec_output1, vec_output2], {vec_input1: test_cap_image, vec_input2: test_db_image})
                        vec1 = np.array(vec1[0])
                        vec2 = np.array(vec2[0])
                        # del vec_session

                        euclidean_distance[idx] = np.sqrt(np.sum(np.square(np.subtract(vec1, vec2))))  # L2 dist

                    print(euclidean_distance)
                    print(np.min(euclidean_distance))

                    # threshold = 141.0
                    if np.min(euclidean_distance) < threshold:
                        print("Pass!")
                        pass_state = True
                    else:
                        print("Fail!")
                        pass_state = False

                        # capture_time = time.time()

            curTime = time.time()
            sec = curTime - preTime
            preTime = curTime
            fps = 1 / sec
            cv2.putText(captured_image, "FPS : %0.1f" % fps, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            height, width, channel = captured_image.shape
            qImg = QtGui.QImage(captured_image.data, width, height, width * channel, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap(qImg)
            pixmap_label.setPixmap(pixmap)

            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                break

        # else:
        #     QtWidgets.QMessageBox.about(win, "Error", "Cannot read frame.")
        #     print("Cannot read frame.")
        #     break

    cap.release()
    cv2.destroyAllWindows()
    print("Thread end.")

def start():
    global running
    if not running:
        running = True
        th = threading.Thread(target=run)
        th.start()
        print("Started...")

def enroll():
    global enrolling, matching
    if not enrolling:
        start()
        enrolling = True
        matching = False
        print("Enrolling...")

def match():
    global enrolling, matching
    if not matching:
        start()
        enrolling = False
        matching = True
        print("Matching...")

def stop():
    global running, enrolling, matching
    if running:
        running = False
        enrolling = False
        matching = False
        print("Stoped...")

def onExit():
    stop()
    print("exit")

if __name__ == "__main__":
    is_periocular = True # 마스크 착용 여부에 따라 결정

    if is_periocular:
        target = "periocular"
    else:
        target = "full_face"

    # load onnx version of BFM
    fd_session = onnxruntime.InferenceSession('onnx/facedetector.onnx', None)
    fd_input_name = fd_session.get_inputs()[0].name
    fd_classes_name = ["BACKGROUND", "FACE"]

    lm_session = onnxruntime.InferenceSession('onnx/TDDFA.onnx', None)
    vec_session = onnxruntime.InferenceSession('./onnx/'+target+'.onnx', None)
    vec_input1 = vec_session.get_inputs()[0].name
    vec_input2 = vec_session.get_inputs()[1].name
    vec_output1 = vec_session.get_outputs()[0].name
    vec_output2 = vec_session.get_outputs()[1].name
    # torch_path = './weights/periocualr epoch-297 loss-0.028618834912776947 th-1.41.pth'
    # vec_session = torch.load(torch_path)  # pkl파일로 학습된 모델을 불러오기

    enrolled_dir = './DB/enrolled_image/'
    os.makedirs(enrolled_dir, exist_ok=True)

    running = False
    enrolling = False
    matching = False
    pass_state = False
    threshold = 141.0

    enrolled_img_list = os.listdir(enrolled_dir)
    euclidean_distance = np.full(len(enrolled_img_list), threshold + 1)

    capture_time= time.time()

    app = QtWidgets.QApplication([])
    win = QtWidgets.QWidget()
    vbox = QtWidgets.QVBoxLayout()
    pixmap_label = QtWidgets.QLabel()
    btn_start = QtWidgets.QPushButton("Camera On")
    btn_enrolling = QtWidgets.QPushButton("Enroll")
    btn_matching = QtWidgets.QPushButton("Match")
    btn_stop = QtWidgets.QPushButton("Camera Off")

    vbox.addWidget(pixmap_label)
    vbox.addWidget(btn_start)
    vbox.addWidget(btn_enrolling)
    vbox.addWidget(btn_matching)
    vbox.addWidget(btn_stop)

    win.setWindowTitle('Masked Face Identifictaion')
    # win.move(0, 0)
    win.resize(664, 650)
    win.setLayout(vbox)
    win.show()

    btn_start.clicked.connect(start)
    btn_enrolling.clicked.connect(enroll)
    btn_matching.clicked.connect(match)
    btn_stop.clicked.connect(stop)
    app.aboutToQuit.connect(onExit)

    sys.exit(app.exec_())
