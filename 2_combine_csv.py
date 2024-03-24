##############################################################################
# 正規化していないlandmarksを後から正規化する際に使う
##############################################################################

# Standard Library
import glob
import csv
import os

# Third-Party Libraries
import pandas as pd


# 正規化処理
def normalize_coordinates(cor_list):
    max_val = max(cor_list)
    min_val = min(cor_list)
    norm_factor = max(abs(max_val), abs(min_val))
    normalized_cor_list = [n / norm_factor for n in cor_list]
    return normalized_cor_list


def write_csv(yubimoji_id, landmark_list, palmlength, csv_path):

    with open(csv_path, 'a', newline="") as file:
        writer = csv.writer(file)

        if palmlength == 0:
            writer.writerow([yubimoji_id, *landmark_list])
        else:
            writer.writerow([yubimoji_id, *landmark_list, palmlength])


def main(import_dir, export_file, buffer_size=15, withPalm=False, palmlengthNormalized=False):

    # 出力先のファイルが存在する場合、削除
    if os.path.exists(export_file):
        os.remove(export_file)

    # パスで指定したファイルの一覧をリスト形式で取得
    csv_files = glob.glob(import_dir)

    # csvファイルの中身を追加していくリストを用意
    data_list = []

    # 読み込むファイルのリストを走査
    for file in csv_files:

        normalized_landmarks_list = []
        palmlength_list = []

        # ファイルを読み込む
        csv_ = pd.read_csv(file, header=None)

        if len(csv_) == buffer_size:
            data_list.append(csv_.tail(buffer_size))
        else:
            print('File size is not enough.')

        landmarks_list = csv_.values.tolist()

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
            normalized_landmarks_list.append(normalized_landmarks)

        # Max-min normalization of 15 flames palmlength within the same file
        if palmlengthNormalized:
            normalized_palmlength_list = normalize_coordinates(palmlength_list)

        for i in range(len(normalized_landmarks_list)):
            if withPalm and palmlengthNormalized:
                write_csv(yubimoji, normalized_landmarks_list[i],
                          normalized_palmlength_list[i], export_file)

            elif withPalm and not palmlengthNormalized:
                write_csv(yubimoji, normalized_landmarks_list[i],
                          palmlength_list[i], export_file)

            else:
                write_csv(
                    yubimoji, normalized_landmarks_list[i], 0, export_file)


# w/o palm
import_dir = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/1_point_history_resized/*.csv'
export_file = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/normalised__point_history_combined.csv'

main(import_dir, export_file)

# w/ palm & w/o palmlengthNormalized
import_dir = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/3_point_history_palm/*.csv'
export_file = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/normalised__point_history_palm_combined.csv'

main(import_dir, export_file, withPalm=True)

# w/ palm & w/ palmlengthNormalized
import_dir = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/3_point_history_palm/*.csv'
export_file = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/normalised__point_history_normalisedpalm_combined.csv'
main(import_dir, export_file, withPalm=True, palmlengthNormalized=True)
