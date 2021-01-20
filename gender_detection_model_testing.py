import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil


class ModelTesting:
    """
    The images to be tested should be of the type:
    name_<<M/F/O>>.jpg
    OR
    name_<<M/F/O>>.jpeg
    """

    def __init__(self, directory_path: str, model_path: str, wrong_class_dir, min_record_writing_interval,
                 outfile_name) -> None:

        self.directory_path = directory_path
        self.wrong_class_dir = wrong_class_dir
        self.model = tf.keras.models.load_model(model_path)
        self.labels = ['F', 'M', 'O']
        self.writing_interval = min_record_writing_interval
        self.outfile_name = outfile_name

    def append_result(self, result_dict: dict, file_name: str, t_v: str, p_v: str, model_res: np.ndarray,
                      type_dict='overall') -> dict:

        result_dict['file'].append(
            file_name if type_dict == 'overall'
            else os.path.join(self.wrong_class_dir, file_name.split(os.sep)[-1])
        )
        result_dict['True Value'].append(t_v)
        result_dict['Predicted Value'].append(p_v)
        result_dict['F%'].append(model_res[0][0])
        result_dict['M%'].append(model_res[0][1])
        result_dict['O%'].append(model_res[0][2])
        return result_dict

    def write_and_reinit_acc(self, res_dict: dict, wrong_dict: dict, mode: str = 'a') -> (dict, dict):

        file_name_res = './outfiles/overall2-%s.csv' % self.outfile_name
        file_name_wrong = './outfiles/wrong2-%s.csv' % self.outfile_name
        df_res = pd.DataFrame(res_dict)
        df_wrong_res = pd.DataFrame(wrong_dict)
        if mode == 'w':
            # df.to_csv('my_csv.csv', mode='a', header=False)
            df_res.to_csv(file_name_res, mode='w', header=True, index=False)
            df_wrong_res.to_csv(file_name_wrong, mode='w', header=True, index=False)
        else:
            df_res.to_csv(file_name_res, mode='a', header=False, index=False)
            df_wrong_res.to_csv(file_name_wrong, mode='a', header=False, index=False)

        res_dict = {'file': [], 'True Value': [], 'Predicted Value': [], 'F%': [], 'M%': [], 'O%': []}
        wrong_dict = {'file': [], 'True Value': [], 'Predicted Value': [], 'F%': [], 'M%': [], 'O%': []}
        return res_dict, wrong_dict

    @staticmethod
    def update_statistics(statistics: dict, t_v: str, p_v: str) -> dict:
        statistics['total_images'] += 1

        # total statistics class-wise
        if t_v == 'M':
            statistics['total_male'] += 1
        elif t_v == 'F':
            statistics['total_female'] += 1
        else:
            statistics['total_others'] += 1

        # correct/incorrect statistics class-wise
        if t_v == p_v:
            statistics['correctly_classified'] += 1
            if t_v == 'M':
                statistics['correct_male'] += 1
            elif t_v == 'F':
                statistics['correct_female'] += 1
            else:
                statistics['correct_others'] += 1
        else:
            statistics['wrongly_classified'] += 1
            if t_v == 'M':
                statistics['misclassified_male'] += 1
            elif t_v == 'F':
                statistics['misclassified_female'] += 1
            else:
                statistics['misclassified_others'] += 1
        print(statistics)
        return statistics

    def test_and_res(self) -> None:
        file_gen = os.scandir(self.directory_path)
        res_dict = {'file': [], 'True Value': [], 'Predicted Value': [], 'F%': [], 'M%': [], 'O%': []}
        wrong_dict = {'file': [], 'True Value': [], 'Predicted Value': [], 'F%': [], 'M%': [], 'O%': []}

        statistics = {
            'total_images': 0,
            'correctly_classified': 0,
            'wrongly_classified': 0,
            'total_male': 0,
            'total_female': 0,
            'total_others': 0,
            'correct_male': 0,
            'correct_female': 0,
            'correct_others': 0,
            'misclassified_male': 0,
            'misclassified_female': 0,
            'misclassified_others': 0
        }

        for i, file in enumerate(file_gen):
            file_path = os.path.join(self.directory_path, file.name)
            img = cv2.imread(file_path)
            resized = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
            resized = resized / 255.0
            img_arr = np.expand_dims(resized, axis=0)

            img_res = self.model.predict(img_arr)

            print(file, img_res)
            t_v = file.name.split('.')[0].split('_')[-1]
            val = np.argmax(img_res, axis=1)
            p_v = self.labels[val[0]]
            statistics = self.update_statistics(statistics, t_v, p_v)

            if t_v != p_v:
                shutil.copy(file_path, self.wrong_class_dir)

                wrong_dict = self.append_result(wrong_dict, file.name, t_v, p_v, img_res, type_dict='wrong')

            res_dict = self.append_result(res_dict, file.name, t_v, p_v, img_res, type_dict='overall')

            print('composite_statistics:', statistics, end='\n')
            if i % self.writing_interval == 0 and i > 0:

                if i / self.writing_interval == 1:
                    res_dict, wrong_dict = self.write_and_reinit_acc(res_dict, wrong_dict, mode='w')
                    print('init-file', i)
                else:
                    print('append_file', i)
                    res_dict, wrong_dict = self.write_and_reinit_acc(res_dict, wrong_dict, mode='a')

        self.write_and_reinit_acc(res_dict, wrong_dict, mode='a')


if __name__ == '__main__':
    mt = ModelTesting('/home/iamtheuserofthis/python_workspace/img_processing/jpeg_images_set/testing_dirs',
                      '/home/iamtheuserofthis/python_workspace/gender_detection/models/gend_detect_without_toons',
                      '/home/iamtheuserofthis/python_workspace/img_processing/jpeg_images_set/wrongly_classified',
                      500,
                      '1x')
    mt.test_and_res()
