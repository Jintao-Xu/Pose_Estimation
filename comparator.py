import numpy as np
import math
from collections import defaultdict

jointCode = {0: 'Head',
        1: 'Neck',
        2: 'Right_Shoulder',
        3: 'Right_Elbow',
        4: 'Right_Wrist',
        5:'Left_Shoulder',
        6:'Left_Elbow',
        7:'Left_Wrist',
        8:'Right_Hip',
        9:'Right_Knee',
        10:'Right_Ankle',
        11:'Left_Hip',
        12:'Left_Knee',
        13:'Left_Ankle',
        14:'Chest',
        15:'Background'}

class Encoder():
    
    def __init__(self, points, speedUp = 0):
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
        
        
    def congregate(self, points, time): #list[list[tuple]] list of frames of points
        for i in range(len(points[0])): #for each joint
            for j in range(len(points)): #get point for each frame
                self.jointFrameMap[jointCode[i]].append(points[j][i])
                
        if time > 0: # adjust speed to match 
            for joint in self.jointFrameMap:
                self.jointFrameMap[joint] = self.adjustSpeed(self.jointFrameMap[joint], time)
                
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
        dx = x2 - x1
        dy = y2 - y1
        return self.codeMap[(dx, dy)]


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
        if(xdif > 0):
            xs = 1
        else:
            xs = -1
        if (ydif > 0):
            ys = 1
        else:
            ys = -1
        if (dx > dy):

            d = 2 * dy - dx
            while (x1 != x2):
                x1 += xs
                if (d >= 0):
                    y1 += ys
                    d -= 2 * dx
                d += 2 * dy
                linePoints.append([x1, y1])
        else:

            d = 2 * dx-dy
            while(y1 != y2):
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
                linePoints = self.applyBresenham(p1[0], p1[1], p2[0], p2[1])
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
        exp_l = len(exp_points)
        act_l = len(act_points)
        if exp_l > act_l:
            speedUp = exp_l / act_l
            self._expect = Encoder(exp_points, speedUp)
            self._actual = Encoder(act_points)
        elif exp_l < act_l:
            speedUp = act_l/exp_l
            self._expect = Encoder(exp_points)
            self._actual = Encoder(act_points, speedUp)
        

    def score(self):
        totalDiff = 0
        maxScore = 100
        for joint in jointCode.values():
            diff = 0
            exp = self._expect.encodedMap[joint]
            act = self._actual.encodedMap[joint]
            totalDiff += abs(len(exp) - len(act)) * 0.1
            # print(exp, act)
            for i in range(min(len(exp), len(act))):
                dissim = self.similarity(int(act[i]), int(exp[i]))
                diff += dissim
                # print(dissim, diff)
            totalDiff += diff
        # print(totalDiff)
        
        #TODO: Do we want to return it as a single score, array of scores, or both?
        # print(maxScore - totalDiff / 15) #Better normalization needed.
        return maxScore - totalDiff / 15
            

    def similarity(self, actual, expected):
        scores = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]
        # print(actual, expected)
        return scores[actual - expected]


actual = [[(347, 125)], [(329, 139)], [(358, 126)], [(376, 140)]]
expected = [[(345, 125)], [(339, 139)], [(357, 126)]]

comp = Comparator(expected, actual)
print(comp.score())
