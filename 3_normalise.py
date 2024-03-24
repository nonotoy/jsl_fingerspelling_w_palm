##############################################################################
# Normalise reshaped landmark coordinates for LSTM training
##############################################################################

# Standard Library
import csv
import os

# Third-Party Libraries
import numpy as np


# 正規化処理
def normalize_coordinates(cor_list):
    max_val = max(cor_list)
    min_val = min(cor_list)
    norm_factor = max(abs(max_val), abs(min_val))
    normalized_cor_list = [n / norm_factor for n in cor_list]
    return normalized_cor_list


# csv保存
def write_csv(landmark_list, yubimoji_id, palmlength, csv_path):

    with open(csv_path, 'a', newline="") as file:
        writer = csv.writer(file)

        if palmlength == 0:
            writer.writerow([yubimoji_id, *landmark_list])
        else:
            writer.writerow([yubimoji_id, *landmark_list, palmlength])


def main(import_file, export_file, withPalm=False, palmlengthNormalized=False):

    if os.path.exists(export_file):
        os.remove(export_file)

    with open(import_file, encoding='utf-8-sig') as f:

        landmarks_list = csv.reader(f)
        palmlength_list = []

        for landmarks in landmarks_list:

            # Fetch yubimoji ID
            yubimoji = landmarks[0]

            if withPalm:
                # Fetch palmlength
                palmlength = float(landmarks[41])
                palmlength_list.append(palmlength)

                # Fetch landmark coordinates
                landmark_tmp = [float(landmarks[i]) for i in range(1, 41)]
            else:
                # Fetch landmark coordinates
                landmark_tmp = [float(i) for i in landmarks[1:]]

            # Max normalization
            normalized_landmarks = normalize_coordinates(landmark_tmp)

            # Write csv
            if withPalm:
                write_csv(normalized_landmarks, yubimoji,
                          palmlength, export_file)

            else:
                write_csv(normalized_landmarks, yubimoji, 0, export_file)

        if palmlengthNormalized:
            normalized_palmlength_list = normalize_coordinates(palmlength_list)
            print(normalized_palmlength_list)


import_file = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/3_point_history_palm copy/point_history_30cm_00_20240119085505.csv'
export_file = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/normalised__point_history_combined_EXP.csv'

main(import_file, export_file, withPalm=True)

'''# w/o palm
import_file = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/point_history_combined.csv'
export_file = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/normalised__point_history_combined.csv'

main(import_file, export_file, withPalm=False)

# w/ palm
import_file = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/point_history_palm_combined.csv'
export_file = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/normalised__point_history_palm_combined.csv'

main(import_file, export_file, withPalm=True)'''
