import numpy as np
import math
from collections import defaultdict

# Same as the dic in movenet_helper
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

jointCount = len(KEYPOINT_DICT)
DELTA = 0.001
THRESHOLD = DELTA / 2

class Encoder():

    def __init__(self, points, speedUp=0):
        self.jointFrameMap = defaultdict(list)

        self.codeMap = {
            (1, 0): 0,
            (1, 1): 1,
            (0, 1): 2,
            (-1, 1): 3,
            (-1, 0): 4,
            (-1, -1): 5,
            (0, -1): 6,
            (1, -1): 7
        }
        self.congregate(points, speedUp)
        self.encodedMap = defaultdict(str)
        self.encode()

    def congregate(self, points, time):  # list[list[tuple]] list of frames of points
        for i,joint in enumerate(KEYPOINT_DICT):  # for each joint
            for j in range(len(points)):  # get point for each frame
                if points[j][i,0:2].any() == None:
                    continue
                self.jointFrameMap[joint].append(points[j][i,0:2])

        if time > 0:  # adjust speed to match
            for joint in self.jointFrameMap:
                self.jointFrameMap[joint] = self.adjustSpeed(self.jointFrameMap[joint], time)
    
    def adjust(encoder1, encoder2):
        for i,joint in enumerate(KEYPOINT_DICT):
            encoder1.encodedMap[joint]
            encoder2.encodedMap[joint]
            l1 = len(encoder1.encodedMap[joint])
            l2 = len(encoder2.encodedMap[joint])
            if l1 > l2:
                speedUp = l1 / l2
                encoder2.extend(joint, speedUp)
            elif l1 < l2:
                speedUp = l2 / l1
                encoder1.extend(joint, speedUp)
    
    def extend(self, joint, ratio):
        old_list = self.encodedMap[joint]
        old_list = [*old_list]
        ratio_int = int(ratio)
        
        
        new_list = [old_list for _ in range(ratio_int)]
        # new_list = "".join(np.array(new_list).transpose().reshape(-1).tolist())
        new_list = np.array(new_list).transpose().reshape(-1).tolist()
        if ratio%1 != 0:
            ratio_fraction = int(1/(ratio%1))
            segment_len = int(len(new_list)/(ratio_fraction+1))
            for i in range(segment_len):
                new_list.insert((segment_len-i-1)*ratio_fraction, new_list[(segment_len-i-1)*ratio_fraction])
        self.encodedMap[joint] = "".join(new_list)

    # def mid_chaincode(ori, des, gap):


    # This function generates the chaincode
    # for transition between two neighbour points
    ''' 
    dx dy    chain  direction
    1  0       0        right
    1  1       1        upper-right
    0  1       2        up
    -1 1       3        upper-left
    -1 0       4        left
    -1 -1      5        lower-left
    0 -1       6        down
    1 -1       7        lower-right
    '''

    def getChainCode(self, x1, y1, x2, y2):
        dx = round(x2 - x1, 3)
        dy = round(y2 - y1, 3)
        return self.codeMap[(int(dx/DELTA), int(dy/DELTA))]

    '''This function generates the list of
    chaincodes for given list of points'''

    def generateChainCode(self, linePoints):
        chainCode = []
        for i in range(len(linePoints) - 1):
            a = linePoints[i]
            b = linePoints[i + 1]
            chainCode.append(self.getChainCode(a[0], a[1], b[0], b[1]))
        return chainCode

    def applyBresenham(self, x1, y1, x2, y2):
        linePoints = []
        linePoints.append([x1, y1])
        xdif = x2 - x1
        ydif = y2 - y1
        dx = abs(xdif)
        dy = abs(ydif)
        if (xdif > 0):
            xs = DELTA
        else:
            xs = -DELTA
        if (ydif > 0):
            ys = DELTA
        else:
            ys = -DELTA
        if (dx > dy):

            d = 2 * dy - dx
            while (abs(x1-x2) > THRESHOLD):
                x1 += xs
                if (d >= 0):
                    y1 += ys
                    d -= 2 * dx
                d += 2 * dy
                linePoints.append([x1, y1])
        else:

            d = 2 * dx - dy
            while (abs(y1 - y2) > THRESHOLD):
                y1 += ys
                if (d >= 0):
                    x1 += xs
                    d -= 2 * dy
                d += 2 * dx
                linePoints.append([x1, y1])
        return linePoints

    def encode(self):
        for joint in self.jointFrameMap:
            jointPoints = self.jointFrameMap[joint]

            totalChain = ""
            for i in range(1, len(jointPoints)):
                p1 = jointPoints[i - 1]
                p2 = jointPoints[i]
                chainCode = ""
                # Refer from movenet_helper l:496, index of x and y
                linePoints = self.applyBresenham(p1[1], p1[0], p2[1], p2[0])
                chainCode = self.generateChainCode(linePoints)
                totalChain += ("".join(str(x) for x in chainCode))
            self.encodedMap[joint] = totalChain

    def adjustSpeed(self, frames, time):
        newData = []
        i = 0
        l = len(frames)

        a = int(time)
        b = math.ceil(time)
        while i < l:
            newData.append(frames[i])
            if (i + a < l) and (i + b) < l:
                spliceFrame = ((frames[int(i + a)], frames[int(i + b)]))
                newData.append(
                    tuple(map(lambda y: int(sum(y) / float(len(y))), zip(*spliceFrame))))

            i += int(time * 2)

        return newData

class Comparator():
    def __init__(self, exp_points, act_points):
        # exp_l = len(exp_points)
        # act_l = len(act_points)
        # if exp_l > act_l:
        #     speedUp = exp_l / act_l
        #     self._expect = Encoder(exp_points, speedUp)
        #     self._actual = Encoder(act_points)
        # elif exp_l < act_l:
        #     speedUp = act_l / exp_l
        #     self._expect = Encoder(exp_points)
        #     self._actual = Encoder(act_points, speedUp)
        # else:
        #     self._expect = Encoder(exp_points)
        #     self._actual = Encoder(act_points)
        self._expect = Encoder(exp_points)
        self._actual = Encoder(act_points)
        Encoder.adjust(self._expect, self._actual)

    def score(self):
        # totalDiff = 0
        maxScore = 100
        jointsScore = {}
        jointsWeight = {}
        for joint in KEYPOINT_DICT:
            exp = self._expect.encodedMap[joint]
            act = self._actual.encodedMap[joint]
            common_len = math.ceil((len(exp) + len(act))/2)
            jointsWeight[joint] = common_len
            mult = maxScore / common_len
            print(joint, len(exp), len(act), len(exp)/len(act))
            diff = abs(len(exp) - len(act)) * 0.01
            # totalDiff += abs(len(exp) - len(act)) * 0.1
            # print(exp, act)
            for i in range(min(len(exp), len(act))):
                # dissim = self.similarity((act[i]), (exp[i]))
                dissim = abs(act[i] - exp[i])
                diff += dissim * mult
                # print(dissim, diff)
            jointsScore[joint] = int(maxScore - diff)
            # totalDiff += diff

        # TODO: Do we want to return it as a single score, array of scores, or both?
        # print(maxScore - totalDiff / 15) #Better normalization needed.
        res = 0
        print(jointsScore)
        for joint in KEYPOINT_DICT:
            res += jointsScore[joint] * jointsWeight[joint] / sum(jointsWeight.values())
            jointsScore[joint] = round(-(100-jointsScore[joint])*jointsWeight[joint] / sum(jointsWeight.values()),3)
        return res, jointsScore
        # return sum(jointsScore.values()) / jointCount, jointsScore

    def similarity(self, actual, expected):
        scores = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]
        # print(actual, expected)
        return scores[actual - expected]

standard_dic = {}
pose_list = ["Push-Up", "Sit-Up", "Squat", "Pull-Up", "Plank", "Jumping-Jack"]
for pose_name in pose_list:
    file = "./standard_pose/" + pose_name + ".txt"
    loaded_kpoints = np.loadtxt(file)
    standard_dic[pose_name] = loaded_kpoints.reshape(
            loaded_kpoints.shape[0], loaded_kpoints.shape[1] // 3, 3)
    
print(standard_dic.keys())

