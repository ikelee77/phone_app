
import socket
import cv2
import numpy as np
import os, sys
from time import strftime, localtime
import random
from openvino.inference_engine import IENetwork, IEPlugin
import time

def get_local_ip():

    ll = [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2]
                        if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)),
                                                              s.getsockname()[0], s.close()) for s in
                                                             [socket.socket(socket.AF_INET,
                                                                            socket.SOCK_DGRAM)]][0][1]]) if l]
    host = ll[0][0]
    return host



#plugin = IEPlugin("MYRIAD", "/opt/intel/openvino_2019.1.144/deployment_tools/inference_engine/lib/intel64")
#model_xml = '/home/intel/my_model/FP16/mobilenet-ssd.xml'
#model_bin = '/home/intel/my_model/FP16/mobilenet-ssd.bin'

plugin = IEPlugin("CPU", "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64")
plugin.add_cpu_extension("/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so")
model_xml = '/home/intel/my_model/FP32/mobilenet-ssd.xml'
model_bin = '/home/intel/my_model/FP32/mobilenet-ssd.bin'


print('Loading network files:\n\t{}\n\t{}'.format(model_xml, model_bin))

net = IENetwork(model=model_xml, weights=model_bin)

net.batch_size = 1
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
exec_net = plugin.load(network=net)

labels = ["plane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "table", "dog", "horse",
                  "motorcycle", "person", "plant", "sheep", "sofa", "train", "monitor"]


cap = cv2.VideoCapture(0)
if not cap.isOpened(): sys.exit('camera error')


sock = socket.socket()
host = get_local_ip()
port = int(sys.argv[1])
sock.bind((host, port))
print("local ip:", host)


while True:

    break_flag = False
    found_count = 0

    sock.listen(1)
    print("listening..")
    conn, addr = sock.accept()
    print("success")

    msg = conn.recv(5)
    target = int(msg.decode())
    prev_target = target

    if target == -1:
        print("target not set")
    else:
        print("looking for target: {}".format(labels[target]))



    while True:

        ret, frame = cap.read()
        if not ret:
            print("capture read fail")
            continue

        rows, cols, channels = frame.shape
        width = cols
        height = rows
        length = min(width, height)
        pt = [60, 60]
        if width < height:
            pt[1] += int((height - length) / 2)
        else:
            pt[0] += int((width - length) / 2)

        length -= 120

        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:
            break

        mid_frame = frame[pt[1]:pt[1] + length, pt[0]:pt[0] + length]
        cut_frame = cv2.resize(mid_frame, (300, 300))
        img = cut_frame
        height, width, _ = img.shape
        n, c, h, w = net.inputs[input_blob].shape
        img2 = img
        if height != h or width != w:
            img2 = cv2.resize(img, (w, h))

        img2 = img2.transpose((2, 0, 1))
        images = np.ndarray(shape=(n, c, h, w))
        images[0] = img2

        res = exec_net.infer(inputs={input_blob: images})
        detections = res[out_blob][0][0]

        for i, detect in enumerate(detections):  # detections: (100, 7)  img_id, label_index, confidence

            image_id = float(detect[0])
            label_index = int(detect[1])
            confidence = float(detect[2])

            if image_id < 0 or confidence == 0.:
                continue

            if confidence > 0.7:

                found_count += 1

                if label_index - 1 == target and found_count > 10:

                    conn.sendall("a".encode())
                    print("found target!")
                    ## save image and start sending


                    filename = "target_" + labels[target] + str(int(time.time())) + ".jpeg"
                    cv2.imwrite(filename, frame)

                    f = open(filename, "rb")
                    buff_size = 255
                    data = f.read(255)

                    while data:
                        conn.sendall(data)
                        data = f.read(255)

                    f.close()
                    conn.close()
                    break_flag = True

                    break

            else:
                found_count = 0

        if break_flag:
            cv2.destroyAllWindows()
            break

        print("sending b")
        conn.sendall("b".encode())
        print("sending b success")
        print("receiving..")
        msg = conn.recv(5)
        target = int(msg.decode())
        print("recv success")
        if target != prev_target:
            print("target changed to {}".format(labels[target]))
            prev_target = target


        cv2.imshow('view', frame)



