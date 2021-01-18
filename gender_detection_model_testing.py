import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from datetime import datetime


class ModelTesting:
    """
    The images to be tested should be of the type:
    name_<<M/F/O>>.jpg
    OR
    name_<<M/F/O>>.jpeg
    """

    def __init__(self, directory_path: str, model_path: str):
        """

        :param directory_path: path to the directory with images in aforementioned format
        :type directory_path: str
        :param model_path: path to the model directory with model stored in .pb format
        :type model_path:
        """
        self.directory_path = directory_path
        self.model = tf.keras.models.load_model(model_path)
        self.labels = ['F', 'M', 'O']

    def test_and_res(self):
        file_gen = os.scandir(self.directory_path)
        res_dict = {'True Value': [], 'Predicted Value': [], 'F%': [], 'M%': [], 'O%': []}
        total_images = 0
        correctly_classified = 0
        wrongly_classified = 0
        for file in file_gen:
            total_images += 1
            file_path = os.path.join(self.directory_path, file.name)
            img = cv2.imread(file_path)
            resized = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
            resized = resized/255.0
            img_arr = np.expand_dims(resized, axis=0)
            print(img_arr.shape)
            img_res = self.model.predict(img_arr)
            print(img_res[0])

            t_v = file.name.split('.')[0].split('_')[-1]
            val = np.argmax(img_res, axis=1)
            p_v = self.labels[val[0]]
            if t_v == p_v:
                correctly_classified += 1
            else:
                wrongly_classified += 1
            res_dict['True Value'].append(t_v)
            res_dict['Predicted Value'].append(p_v)
            res_dict['F%'].append(img_res[0][0])
            res_dict['M%'].append(img_res[0][1])
            res_dict['O%'].append(img_res[0][2])

        df = pd.DataFrame(res_dict)
        print(total_images, correctly_classified, wrongly_classified)
        print(df.describe())
        df.to_csv(path_or_buf='./outfiles/%s.csv' % round(datetime.timestamp(datetime.now())), index=False)


if __name__ == '__main__':
    mt = ModelTesting('/home/iamtheuserofthis/python_workspace/img_processing/jpeg_images_set/dry_test_imgs',
                      '/home/iamtheuserofthis/python_workspace/gender_detection/models/gend_detect_without_toons')
    mt.test_and_res()
