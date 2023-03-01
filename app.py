import os

from flask import Flask, request, jsonify, send_file
import base64
import io
import flask_monitoringdashboard as dashboard
import cv2
import numpy as np
import time
from comparator import Comparator, standard_dic
from movenet_helper import *

app = Flask(__name__)
dashboard.bind(app)


@app.route("/feedback", methods=["POST"])
def create_feedback():
    global global_index
    encoded_string = request.json["video"]
    mode = request.json["mode"]
    standard = standard_dic[mode]
    print("Recieve request, staring processing: ", mode)

    with open("input.mp4", "wb") as fh:
        fh.write(base64.b64decode(encoded_string))

    # Specify the paths for the 2 files
    protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose_iter_160000.caffemodel"

    # Read the network into Memory
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    
    inWidth = 368
    inHeight = 368
    threshold = 0.1

    start_time = time.time()

    input_source = "input.mp4"
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()

    image_height, image_width, _ = frame.shape
    crop_region = init_crop_region(image_height, image_width)

    output_images = []
    kpoints = []
    while hasFrame:
        keypoints_with_scores = run_inference(
            movenet, frame, crop_region,
            crop_size=[input_size, input_size])
        frame_drawn = draw_prediction_on_image(
            frame,
            keypoints_with_scores, crop_region=None,
            close_figure=True, output_image_height=300)
        output_images.append(frame_drawn)
        crop_region = determine_crop_region(
            keypoints_with_scores, image_height, image_width)
        kpoints.append(keypoints_with_scores)
        hasFrame, frame = cap.read()
        hasFrame, frame = cap.read()
        hasFrame, frame = cap.read()
    print("Modelling time: ", time.time() - start_time)
    # Prepare gif visualization.
    output = np.stack(output_images, axis=0)
    output_path = "user_result/output" + str(global_index)
    global_index += 1
    to_mp4(output, fps=5, name=output_path)
    #TODO: suppost to compare with standard
    kpoints = np.array(kpoints)
    kpoints = kpoints.reshape(kpoints.shape[0], 17, 3)
    comp = Comparator(standard, kpoints)
    totalScore, jointsScore = comp.score()
    print("Comparation time: ", time.time() - start_time)
    print(jointsScore)
    worstThree = sorted(jointsScore.items(), key=lambda pair: pair[1])[0:3]
    scores = {}
    for pair in worstThree:
        scores[pair[0]] = pair[1]
    print("score: ", totalScore)

    if os.path.exists("output2.mp4"):
        os.remove("output2.mp4")
    os.system("ffmpeg -i ./"+ output_path+".mp4 -vcodec libx264 output2.mp4")

    with open('output2.mp4', 'rb') as f:
        output = f.read()

    # return send_file(io.BytesIO(output), mimetype="video/x-msvideo")
    encoded_string = base64.b64encode(output).decode()
    print("Total execution time: ", time.time() - start_time)
    return jsonify({
        "video": encoded_string,
        "total": int(totalScore),
        "scores": scores
    })


@app.route("/test", methods=["GET", "POST"])
def test():
    return jsonify({
        "score": 100
    })

global_index = 0
if __name__ == "__main__":
    app.run(debug=True)
