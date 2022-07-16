import numpy as np
import math
from collections import defaultdict

class Comparator():
    
    def __init__(self):
        self.jointFrameMap = defaultdict(list)
        self.jointCode = {0: 'Head',
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
        
        
    def congregate(self, points): #list[list[tuple]] list of frames of points
        for i in range(len(points[0])):
            for j in range(len(points)):
                self.jointFrameMap[self.jointCode[i]].append(points[j][i])
                
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
    
    def applyBresenham(x1, y1, x2, y2):
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
    
def main():
    comparator = Comparator()
    
if __name__ == 'main':
    main()
