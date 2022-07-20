from flask import Flask, request, jsonify, send_file
import base64
import io
import flask_monitoringdashboard as dashboard
import cv2
import numpy as np
import time
from comparator import Comparator

app = Flask(__name__)
dashboard.bind(app)

squat_standard = [[(695, 140), (667, 187), (667, 219), (723, 281), (723, 234), (640, 203), (584, 281), (584, 234), (667, 344), (695, 469), (723, 610), (640, 344), (640, 453), (612, 563), (640, 281)], [(695, 140), (667, 187), (667, 219), (723, 281), (723, 234), (640, 203), (584, 281), (584, 234), (667, 344), (695, 469), (723, 610), (640, 344), (640, 453), (612, 563), (640, 281)], [(723, 140), (695, 187), (695, 219), (723, 281), (751, 219), (640, 203), (584, 281), (640, 234), (667, 344), (695, 469), (723, 610), (640, 344), (640, 453), (612, 563), (667, 281)], [(723, 140), (667, 187), (695, 219), (723, 281), (751, 219), (640, 203), (584, 266), (640, 234), (667, 344), (695, 453), (723, 610), (640, 344), (640, 453), (612, 563), (667, 281)], [(723, 140), (695, 187), (695, 219), (723, 281), (751, 219), (640, 203), (584, 266), (640, 234), (667, 344), (695, 453), (723, 610), (640, 344), (640, 453), (612, 563), (640, 281)], [(723, 140), (667, 203), (695, 219), (723, 281), (751, 234), (640, 219), (584, 281), (640, 234), (640, 344), (695, 453), (723, 610), (612, 344), (612, 453), (612, 563), (667, 281)], [(723, 140), (695, 203), (695, 234), (723, 297), (751, 234), (640, 219), (584, 281), (640, 250), (640, 360), (695, 469), (723, 610), (612, 360), (612, 469), (612, 563), (640, 297)], [(723, 156), (667, 219), (695, 234), (723, 297), (751, 250), (640, 234), (584, 297), (640, 266), (640, 375), (723, 469), (723, 594), (612, 360), (612, 469), (612, 563), (640, 297)], [(723, 187), (695, 234), (695, 266), (723, 328), (751, 266), (640, 250), (584, 313), (612, 281), (640, 391), (723, 485), (723, 610), (612, 375), (612, 469), (612, 563), (640, 328)], [(723, 187), (695, 250), (695, 281), (723, 344), (751, 281), (640, 266), (584, 328), (584, 281), (640, 406), (723, 485), (723, 610), (612, 406), (612, 485), (612, 563), (640, 344)], [(723, 219), (695, 281), (695, 297), (723, 360), (751, 313), (640, 297), (584, 344), (584, 313), (640, 422), (751, 500), (723, 610), (612, 422), (612, 485), (612, 563), (640, 360)], [(723, 234), (695, 297), (695, 313), (723, 391), (751, 328), (640, 313), (584, 360), (640, 328), (640, 438), (751, 500), (723, 610), (612, 438), (751, 500), (612, 563), (640, 375)], [(723, 250), (695, 313), (695, 328), (723, 406), (751, 344), (640, 328), (584, 375), (640, 344), (640, 453), (751, 500), (723, 610), (612, 453), (612, 547), (612, 563), (667, 391)], [(723, 266), (695, 328), (695, 344), (723, 422), (751, 360), (640, 344), (584, 391), (584, 344), (640, 469), (751, 500), (723, 610), (612, 453), (612, 516), (612, 563), (667, 406)], [(723, 281), (667, 344), (695, 360), (723, 422), (751, 375), (640, 344), (584, 406), (612, 375), (640, 469), (751, 500), (723, 610), (612, 469), (612, 532), (612, 563), (667, 406)], [(723, 281), (695, 344), (695, 360), (723, 438), (751, 375), (640, 360), (584, 406), (612, 375), (640, 485), (751, 500), (723, 610), (612, 469), (612, 532), (612, 563), (667, 422)], [(723, 281), (667, 344), (695, 360), (723, 422), (751, 375), (640, 360), (584, 406), (612, 375), (640, 469), (751, 500), (723, 610), (612, 469), (612, 516), (612, 563), (667, 422)], [(723, 266), (695, 313), (695, 344), (723, 406), (751, 360), (640, 328), (584, 391), (612, 360), (640, 453), (751, 500), (723, 610), (612, 453), (640, 516), (612, 563), (667, 391)], [(723, 219), (695, 281), (695, 297), (723, 360), (751, 313), (640, 281), (584, 344), (612, 313), (640, 422), (779, 500), (723, 610), (612, 422), (612, 485), (612, 563), (667, 360)], [(723, 172), (695, 234), (695, 266), (723, 328), (751, 266), (640, 250), (584, 313), (640, 281), (640, 375), (723, 485), (723, 610), (612, 375), (612, 469), (612, 563), (667, 313)], [(695, 140), (695, 203), (695, 234), (723, 297), (751, 234), (640, 219), (584, 281), (640, 250), (640, 360), (695, 469), (723, 610), (612, 360), (612, 453), (723, 610), (667, 297)], [(723, 140), (667, 203), (695, 219), (723, 281), (751, 234), (640, 219), (584, 281), (640, 250), (667, 344), (695, 438), (723, 610), (640, 344), (640, 453), (612, 563), (667, 281)], [(723, 140), (695, 187), (695, 219), (723, 281), (751, 219), (640, 203), (584, 281), (640, 234), (667, 344), (695, 453), (723, 610), (640, 344), (640, 453), (612, 563), (667, 281)], [(695, 125), (667, 187), (695, 219), (723, 281), (751, 219), (640, 203), (584, 281), (640, 234), (667, 344), (695, 453), (723, 610), (640, 344), (640, 453), (612, 563), (667, 281)], [(695, 125), (667, 187), (695, 219), (723, 281), (751, 219), (640, 203), (584, 281), (640, 250), (667, 344), (695, 453), (723, 610), (640, 344), (640, 453), (612, 563), (667, 281)], [(695, 125), (667, 187), (695, 219), (723, 281), (751, 219), (640, 203), (584, 281), (612, 250), (667, 344), (695, 453), (723, 610), (640, 344), (640, 453), (612, 563), (640, 281)]]

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

    all_frame_joints = []
    
    while time.time() - start_time < 60:
        print(time.time() - start_time)
        # t = time.time()
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

        all_frame_joints.append(points)
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

    comp = Comparator(squat_standard, all_frame_joints)
    score = comp.score()
    
    with open('output.avi', 'rb') as f:
        output = f.read()
        
    # return send_file(io.BytesIO(output), mimetype="video/x-msvideo")
    encoded_string = base64.b64encode(output).decode()
    return jsonify({
        "video": encoded_string,
        "score": score
    })


@app.route("/test", methods=["POST"])
def test():
    return jsonify({
        "score": 100
    })

if __name__ == "__main__":
    app.run(debug=True)
