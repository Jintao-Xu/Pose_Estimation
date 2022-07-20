from flask import Flask, request, jsonify, send_file
import base64
import io
import flask_monitoringdashboard as dashboard
import cv2
import numpy as np
import time

app = Flask(__name__)
dashboard.bind(app)

@app.route("/feedback", methods=["POST"])
def create_feedback():
    # encoded_tring = request.json["video"]
    # with open("input.mp4", "wb") as fh:
    #     fh.write(base64.b64decode(encoded_tring))

    # Specify the paths for the 2 files
    protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose_iter_160000.caffemodel"

    # Read the network into Memory
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    
    inWidth = 368
    inHeight = 368
    threshold = 0.1


    input_source = "input.mp4"
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()
    
    # vid_writer = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))
    vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
    
    start_time = time.time()
    
    start_time = time.time()

    while True:
        print(time.time() - start_time)
        hasFrame, frame = cap.read()
        hasFrame, frame = cap.read()
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            break
        
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold : 
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else :
                points.append(None)
        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        # cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.imshow('Output-Keypoints', frameCopy)
        # cv2.imshow('Output-Skeleton', frame)

        vid_writer.write(frame)

    vid_writer.release()
    
    with open('output.avi', 'rb') as f:
        output = f.read()
        
    # return send_file(io.BytesIO(output), mimetype="video/x-msvideo")
    encoded_string = base64.b64encode(output).decode()
    return jsonify({
        "video": encoded_string,
        "score": 100
    })


@app.route("/test", methods=["POST"])
def test():
    return jsonify({
        "score": 100
    })

if __name__ == "__main__":
    app.run(debug=True)
