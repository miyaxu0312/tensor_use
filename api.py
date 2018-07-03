import numpy as np
import os
from predictor import PosPrediction

class PRN:
    ''' Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
    Args:
        is_dlib(bool, optional): If true, dlib is used for detecting faces.
        prefix(str, optional): If run at another folder, the absolute path is needed to load the data.
    '''
    def __init__(self, is_dlib = False, prefix = '.'):

        # resolution of input and output image size.
        self.resolution_inp = 256
        self.resolution_op = 256

        #---- load PRN 
        self.pos_predictor = PosPrediction(self.resolution_inp, self.resolution_op)
        prn_path = os.path.join(prefix, 'Data/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(prn_path + '.data-00000-of-00001'):
            print("please download PRN trained model first.")
            exit()
        self.pos_predictor.restore(prn_path)


    def net_forward(self, image):
        ''' The core of out method: regress the position map of a given image.
        Args:
            image: (256,256,3) array. value range: 0~1
        Returns:
            pos: the 3D position map. (256, 256, 3) array.
        '''
        return self.pos_predictor.predict(image)









