##############################################################
# yubimoji.pyで取得したランドマークのフレーム数の調整
##############################################################

# Standard Library
import csv
import glob
import math
import os
import re

# Third-Party Libraries
import numpy as np

sta_short_frame_files = []
dyn_short_frame_files = []
subject_to_downsample_files = []
dyn_error_files = []
files_tobetransferred = []
processed_files = []


class LandmarkProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.yubimoji, self.landmarks_list = self.convert_csv_to_list()

    def convert_csv_to_list(self):
        with open(self.csv_path, encoding='utf-8-sig') as f:
            landmarks_list = list(csv.reader(f))

            landmarks_list_copy = []

            for landmarks in landmarks_list:
                landmark_tmp = []

                # 1列目 -> 指文字IDを取得
                # それ以降はランドマークを取得
                for i in range(len(landmarks)):
                    if i == 0:
                        yubimoji = int(landmarks[i])
                    else:
                        landmark_tmp.append(float(landmarks[i]))

                landmarks_list_copy.append(landmark_tmp)

            return yubimoji, landmarks_list_copy

    @staticmethod
    def detect_movement(lm_a, lm_b):
        # ランドマーク間の距離を計算する関数

        sum_distance = 0
        lm_count = 21

        for i in range(lm_count):

            if i % 2 == 1:
                continue

            x_a = lm_a[i]
            x_b = lm_b[i]
            y_a = lm_a[i + 1]
            y_b = lm_b[i + 1]

            sum_distance += np.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)

        return sum_distance / lm_count

    @staticmethod
    def calc_palm_length(landmarks):

        # 手掌長 (手首と中指第一関節間の距離) の取得 (Priyaら (2023))
        distance_x = -landmarks[16]  # 0番目のx座標 (= 0) - 9番目のx座標
        distance_y = -landmarks[17]  # 0番目のx座標 (= 0) - 9番目のy座標
        palmLength = math.sqrt(distance_x ** 2 + distance_y ** 2)
        return palmLength


class FrameAnalyzer:
    def __init__(self, frame_size, movement_threshold, is_check_mode=False):
        self.frame_size = frame_size
        self.movement_threshold = movement_threshold
        self.is_check_mode = is_check_mode

    def compare_frame_movement(self, landmarks_list, frame_focus=5, is_stop_detection=False):

        for i in range(len(landmarks_list) - 1):

            nomovement_cnt = 0

            if i < frame_focus - 1:
                continue

            # 過去5フレーム分のランドマーク間の距離を計算
            for frame_ctrl in range(frame_focus):
                distance = LandmarkProcessor.detect_movement(
                    landmarks_list[i - frame_ctrl], landmarks_list[i + 1 - frame_ctrl])
                if distance <= self.movement_threshold:
                    nomovement_cnt += 1

            if nomovement_cnt >= frame_focus:
                if is_stop_detection:
                    fn_stabilised = i - frame_focus + 1
                    return fn_stabilised
            else:
                if not is_stop_detection:
                    return i

        return None

    def compare_palm_length(self, landmarks_list, dif_threshold=20.0, frame_focus=5):

        # 手掌長比較

        depth_change_detected_num = None
        depth_change_stopped_num = None

        for i in range(len(landmarks_list) - 1):

            detected_marker = ''

            if i < frame_focus:
                continue

            palm_length_cur = LandmarkProcessor.calc_palm_length(
                landmarks_list[i])
            palm_length_comp = LandmarkProcessor.calc_palm_length(
                landmarks_list[i - frame_focus])

            palm_length_dif = (palm_length_cur - palm_length_comp) ** 2

            if palm_length_dif >= dif_threshold:
                detected_marker = '***'

                # 初回動作検知時、もしくは連続動作検知時の場合は、フレーム番号を更新
                if depth_change_detected_num == None:
                    depth_change_detected_num = i - frame_focus
                    depth_change_stopped_num = i

                if depth_change_detected_num != None and depth_change_stopped_num != None:
                    depth_change_stopped_num = i

            if self.is_check_mode:
                print('\t', i, palm_length_cur)
                print('\t', i - frame_focus, palm_length_comp)
                print('\t', palm_length_dif, detected_marker)
                print('='*20)

        return depth_change_detected_num, depth_change_stopped_num

    def interpolate_frame(self, landmarks_list):

        new_landmark_array = np.array(landmarks_list)

        # フレーム数とデータ点の数を取得
        num_frames, num_points = new_landmark_array.shape

        # 目標とするフレーム数
        target_frames = self.frame_size

        # 新しいフレームを格納する配列
        frame_interpolated = np.zeros((target_frames, num_points))

        # interpolate
        for i in range(num_points):
            frame_interpolated[:, i] = np.interp(
                np.linspace(0, num_frames - 1, target_frames),
                np.arange(num_frames),
                new_landmark_array[:, i]
            )

        interpolated_list = frame_interpolated.tolist()

        if self.is_check_mode:
            print('='*20)

            # x座標で比較
            cln = 0
            for row in range(len(landmarks_list)):
                print(landmarks_list[row][cln])

            print('='*20)

            for row_new in range(len(interpolated_list)):
                print(interpolated_list[row_new][cln])

            print('Length of interpolated frames:', len(interpolated_list))

        return frame_interpolated.tolist()

    def adjust_frame(self, landmarks_list, file, is_static=False):

        if 1 < len(landmarks_list) < self.frame_size:
            # フレーム数が足りない場合、線形補間によるフレーム補填処理へ
            processed_files.append(file)
            landmarks_list_new = self.interpolate_frame(landmarks_list)
            return landmarks_list_new

        elif len(landmarks_list) <= 1:
            # フレームが1個以下の場合は、補填処理を行わず無視
            sta_short_frame_files.append(
                file) if is_static else dyn_short_frame_files.append(file)
            return -1

        elif len(landmarks_list) > self.frame_size:
            # フレーム数が多い場合はダウンサンプリングを実施 - しない
            subject_to_downsample_files.append(file)
            return -1

        else:
            # フレーム数がピッタリ設定数の場合は、そのまま
            processed_files.append(file)
            return landmarks_list


class FileManager:
    def __init__(self):
        self.label_file_path = 'setting/hand_keypoint_classifier_label.csv'
        self.point_history_path = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/0_point_history'

    def count_files(self, distance_list):
        with open(self.label_file_path, encoding='utf-8-sig') as f:
            yubimoji_labels = [row[0] for row in csv.reader(f)]

        files = os.listdir(self.point_history_path)
        re_record_yubimoji = []
        nec_file_count = 0

        for i, label in enumerate(yubimoji_labels):
            if i > 75:
                break
            for dist in distance_list:
                pattern = f'point_history_{dist}_{i:02}_'
                count = sum(1 for file in files if re.match(pattern, file))
                if count != 10:
                    nec_file_count += 10 - count
                    re_record_yubimoji.append(
                        [f'{i}_{label}', [dist, 10 - count]])

        return sorted(re_record_yubimoji, key=lambda x: x[1], reverse=True), nec_file_count

    def print_frame_info(self, landmarks_list, yubimoji, fn_stabilised, fn_movement_detected, fn_movement_stopped, new_frame_length):
        # 補填前のフレーム数を確認するための関数

        original_landmarks_length = len(landmarks_list)

        for i in range(len(landmarks_list) - 1):

            specStr = f'|| 手掌長: ' + \
                format(LandmarkProcessor.calc_palm_length(
                    landmarks_list[i]), '0.4f')

            if i == fn_stabilised:
                specStr += ' << 初期動作収束'
            if i == fn_movement_detected:
                specStr += ' << 動作検知: 取得範囲 - 自'
            elif i == fn_movement_stopped:
                specStr += ' << 動作停止: 取得範囲 - 至'

            print(i, '\t次フレームとの誤差', format(LandmarkProcessor.detect_movement(
                landmarks_list[i], landmarks_list[i + 1]), '0.4f'), specStr)

        print('------------')
        print('指文字:\t\t\t', yubimoji)
        print('当初のフレーム数:\t', original_landmarks_length)
        print('初期動作収束箇所:\t', fn_stabilised)

        if yubimoji in [24, 34, 39] or yubimoji in range(44, 76):
            # Dynamic yubimoji
            print('動作開始箇所:\t\t', fn_movement_detected)

        else:
            # Static yubimoji
            print('動作開始箇所:\t\t', fn_stabilised)

        print('動作停止箇所:\t\t', fn_movement_stopped)
        print('補填前フレーム数:\t', new_frame_length)


class YubimojiAnalyzer:
    def __init__(self, csv_files, frame_size):
        self.csv_files = csv_files
        self.frame_size = frame_size
        self.movement_threshold = 2.0
        self.file_manager = FileManager()

    def analyze(self, is_check_mode=False, reshape_enable=False, file_transfer_enable=False, reshape_palmlen_enable=False):

        self.is_check_mode = is_check_mode
        self.reshape_enable = reshape_enable
        self.reshape_palmlen_enable = reshape_palmlen_enable
        self.file_transfer_enable = file_transfer_enable

        self.frame_analyzer = FrameAnalyzer(
            self.frame_size, self.movement_threshold, self.is_check_mode)

        # 確認用変数・リスト
        dyn_cnt = 0
        sta_cnt = 0
        exceptfiles = []
        sta_short_frame_files = []
        dyn_short_frame_files = []
        subject_to_downsample_files = []
        dyn_error_files = []
        files_tobetransferred = []
        self.processed_files = []

        # 読み込むファイルのリストを走査
        for file in self.csv_files:

            # 格納用リストの初期化
            new_landmark_list = []

            # csvから指文字IDと加工前ランドマークを取得
            landmark_data = LandmarkProcessor(file)
            yubimoji, landmarks_list = landmark_data.yubimoji, landmark_data.landmarks_list

            # 初期動作が収束したフレーム番号を検出 / 収束前フレームは消去
            fn_stabilised = self.frame_analyzer.compare_frame_movement(
                landmarks_list, is_stop_detection=True)

            if fn_stabilised is not None:

                # 初期動作が収束以降のデータを取得
                landmarks_list = landmarks_list[fn_stabilised:]

                # 動的指文字 - の、も、り、を、ん、濁音、半濁音、促音、拗音
                if yubimoji in [24, 34, 39] or yubimoji in range(44, 76):

                    dyn_cnt += 1
                    is_static = False

                    # 動作開始フレームの番号を取得
                    fn_flat_movement_detected = self.frame_analyzer.compare_frame_movement(
                        landmarks_list)

                    # 動的指文字の場合は、後ろからフレームを走査し、動作停止フレームを検出
                    fn_flat_movement_stopped = self.frame_analyzer.compare_frame_movement(
                        landmarks_list[::-1])

                    # 前後動作検知
                    if yubimoji in [45, 71, 72, 73, 74, 75]:
                        fn_depth_movement_detected, fn_depth_movement_stopped = self.frame_analyzer.compare_palm_length(
                            landmarks_list)

                    # x軸、y軸の動きを検知し、かつその動作の停止を確認している場合
                    if fn_flat_movement_stopped is not None and fn_flat_movement_detected is not None:

                        frame_until = len(landmarks_list) - \
                            fn_flat_movement_stopped + 2

                        if frame_until <= fn_flat_movement_detected:
                            dyn_error_files.append(file)
                            continue

                        # 動作開始の一つ前のフレームから動作停止までのフレームを取得
                        landmarks_list_copy = landmarks_list[fn_flat_movement_detected+1:frame_until]

                        # 確認用
                        if self.is_check_mode:
                            self.file_manager.print_frame_info(
                                landmarks_list,
                                yubimoji,
                                fn_stabilised,
                                fn_flat_movement_detected,
                                fn_flat_movement_stopped,
                                len(landmarks_list_copy)
                            )

                    # 奥行き方向の動作がある指文字で、かつz軸の動き (手掌長の変化) が検知し、かつその動作の停止を確認している場合
                    elif fn_depth_movement_detected is not None and yubimoji in [45, 71, 72, 73, 74, 75]:

                        landmarks_list_copy = landmarks_list[
                            fn_depth_movement_detected:fn_depth_movement_stopped+1]

                        # 確認用
                        if self.is_check_mode:
                            self.file_manager.print_frame_info(
                                landmarks_list,
                                yubimoji,
                                fn_stabilised,
                                fn_depth_movement_detected,
                                fn_depth_movement_stopped,
                                len(landmarks_list_copy)
                            )

                    # 動作開始フレームが検出できなかった場合、動作停止を検出できなかった場合
                    else:
                        '''if isCheckMode:
                            for i in range(len(landmarks_list) - 1):
                                print(i, '\t', '{:.2g}'.format(detectMovement(
                                    landmarks_list[i], landmarks_list[i + 1])), '\t', '{:.5g}'.format(calc_PalmLength(landmarks_list[i])))'''
                        dyn_error_files.append(file)
                        break

                # 静的指文字
                else:
                    sta_cnt += 1
                    is_static = True

                    landmarks_list_copy = landmarks_list[:self.frame_size]

                    # 確認用
                    if self.is_check_mode:
                        self.file_manager.print_frame_info(
                            landmarks_list,
                            yubimoji,
                            fn_stabilised,
                            None,
                            self.frame_size,
                            len(landmarks_list_copy)
                        )
                    # --------------------------------------------

                new_landmark_list = self.frame_analyzer.adjust_frame(
                    landmarks_list_copy, file, is_static=is_static)

                if new_landmark_list == -1:
                    break

                # --------------------------------------------
                # 編集中: 手掌長の取得
                new_landmark_list_palm = new_landmark_list.copy()
                for i in range(len(new_landmark_list_palm)):
                    palm_length = LandmarkProcessor.calc_palm_length(
                        new_landmark_list_palm[i])
                    new_landmark_list_palm[i].extend([palm_length])
                # --------------------------------------------

            # No motion stop detected - Exclude the file
            else:
                exceptfiles.append(file)

            # Write the interpolated landmarks to a new csv file
            if self.reshape_enable and isinstance(new_landmark_list, list) and len(new_landmark_list) == self.frame_size:

                # csv_pathのファイルが存在する場合、削除
                csv_path = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/1_point_history_resized/' + \
                    os.path.basename(file)

                if os.path.exists(csv_path):
                    os.remove(csv_path)

                # 処理後のリストをcsvに書き込む
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in new_landmark_list:
                        writer.writerow([yubimoji] + row)

            # Write the interpolated landmarks with palm length to a new csv file
            if self.reshape_palmlen_enable and isinstance(new_landmark_list_palm, list) and len(new_landmark_list_palm) == self.frame_size:

                # csv_pathのファイルが存在する場合、削除
                csv_path = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/3_point_history_palm/' + \
                    os.path.basename(file)

                if os.path.exists(csv_path):
                    os.remove(csv_path)

                # 処理後のリストをcsvに書き込む
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for row in new_landmark_list:
                        writer.writerow([yubimoji] + row)

        files_tobetransferred.extend(dyn_error_files)
        files_tobetransferred.extend(subject_to_downsample_files)
        files_tobetransferred.extend(dyn_short_frame_files)
        files_tobetransferred.extend(sta_short_frame_files)
        files_tobetransferred.extend(exceptfiles)

        # 処理不可ファイルを指定フォルダへ移動
        if self.file_transfer_enable:

            dir_name = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/2_point_history_check'

            for file in files_tobetransferred:
                os.rename(file, dir_name + '/' + os.path.basename(file))

            if len(files_tobetransferred) > 0:
                print(
                    '* More than 1 file(s) are moved to {0}. Check them and re-record.'.format(dir_name))

        # Summary
        print('##############################################################')
        print('dyn: Files with no motion & motion-stop detected:',
              '\n\t'.join(dyn_error_files))
        print('dyn: Files to be downsampled:',
              '\n\t'.join(subject_to_downsample_files))
        print('dyn: Files with insufficient frames:',
              '\n\t'.join(dyn_short_frame_files))
        print('sta: Files with insufficient frames:',
              '\n\t'.join(sta_short_frame_files))
        print('error: No motion stop detected:',
              '\n\t'.join(exceptfiles))
        print('##############################################################\n')

        print('##############################################################')
        print('All files:', len(self.csv_files))
        print(' - Dynamic yubimoji files:', dyn_cnt)
        print('   - Successfully processed:', dyn_cnt - len(subject_to_downsample_files) -
              len(dyn_error_files) - len(dyn_short_frame_files))
        print('   - Files with no motion & motion-stop detected:',
              len(dyn_error_files))
        print('   - Files to be downsampled:',
              len(subject_to_downsample_files))
        print('   - Files with insufficient frames:', len(dyn_short_frame_files))
        print(' - Static yubimoji files:', sta_cnt)
        print('   - Successfully processed:',
              sta_cnt - len(sta_short_frame_files))
        print('   - Files with insufficient frames:', len(sta_short_frame_files))
        print(' - Error files', len(exceptfiles))
        print('##############################################################\n')

        print('##############################################################')

        # Count necessary files
        distance_list = ['30cm', '50cm', '70cm']
        sorted_list, total_necessary_files = self.file_manager.count_files(
            distance_list)

        print('Necessary files:')
        for row in sorted_list:
            print(row)
        print("Total necessary files:", total_necessary_files)
        print('##############################################################')

# Main function


# Directories where csv files are stored
csv_files = glob.glob(
    '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/0_point_history/*.csv')

analyzer = YubimojiAnalyzer(csv_files, frame_size=15)
analyzer.analyze(is_check_mode=False,
                 reshape_enable=True,
                 file_transfer_enable=True,
                 reshape_palmlen_enable=True)
