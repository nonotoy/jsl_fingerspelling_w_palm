##############################################################################
# Normalise reshaped landmark coordinates for LSTM training
##############################################################################

# Standard Library
import csv
import math
import glob
import datetime
import os


# yubimoji.calcの同名関数と同じだが、入力するランドマークがリスト形式
# z軸の情報が元々出力されていないので、除外する為のフラグは引数として入れていない
def calc_PalmLength(landmarks):
    # 手掌長 (手首と中指第一関節間の距離) の取得 (Priyaら (2023))

    # 手首の座標を取得
    lm_0_x = landmarks[0]  # 0
    lm_0_y = landmarks[1]  # 0

    # 中指第一関節の座標を取得
    lm_9_x = landmarks[16]
    lm_9_y = landmarks[17]

    # 中指第一関節 - 手首の各軸の距離を計算
    distance_x = -lm_9_x
    distance_y = -lm_9_y

    # 中指第一関節 - 手首の距離を計算
    palmLength = math.sqrt(distance_x ** 2 + distance_y ** 2)

    return palmLength


def normalize_coordinates(landmark_list, palmLength=None):

    norm_factor = max(
        abs(n) for n in landmark_list) if palmLength == None else palmLength

    # 正規化処理
    normalized_landmark_list = [n / norm_factor for n in landmark_list]
    return normalized_landmark_list


def write_csv(landmark_list, yubimoji_id, csv_path):
    # csv保存

    with open(csv_path, 'a', newline="") as file:
        writer = csv.writer(file)
        writer.writerow([yubimoji_id, *landmark_list])


def main(isPalmNormalised=False):

    parent_dir = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/'

    file = 'point_history/point_history_combined.csv'

    if isPalmNormalised:
        print('Normalised by palm length')
        csv_path = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/point_history_combined_palmnormalised.csv'
    else:
        csv_path = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/point_history_combined_normalised.csv'

    import_file = parent_dir + file
    export_file = parent_dir + csv_path

    if os.path.exists(export_file):
        os.remove(export_file)

    with open(import_file, encoding='utf-8-sig') as f:

        landmarks_list = csv.reader(f)

        for landmarks in landmarks_list:

            landmark_tmp = []
            palmLength = None

            # 一列目から指文字IDを取得、それ以降はランドマークを取得し、リストに格納
            # 指文字が取れてない。reshape前で落としてしまっているので、直す。
            for i in range(len(landmarks)):
                if i == 0:
                    yubimoji = landmarks[i]
                else:
                    landmark_tmp.append(float(landmarks[i]))

            # 純粋な正規化処理のみ、相対座標云々はランドマーク取得時に実施済み
            if isPalmNormalised:
                palmLength = calc_PalmLength(landmark_tmp)
                normalized_landmarks = normalize_coordinates(
                    landmark_tmp, palmLength)

            else:
                normalized_landmarks = normalize_coordinates(landmark_tmp)

            # 書き出し
            write_csv(normalized_landmarks, yubimoji, export_file)


# main()
main(isPalmNormalised=True)
