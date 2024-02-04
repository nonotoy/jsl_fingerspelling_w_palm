##############################################################################
# Description: 日本手話の指文字を認識するプログラム
##############################################################################

# Standard Library
import os
import datetime
import time
import csv
import math
import copy
import itertools

# Third-Party Libraries
import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

# Local Libraries
from model import KeyPointClassifier as KeyPointClassifier
from palm_model import KeyPointClassifier as KeyPointClassifierwithPalm


class HandGestureRecognition:
    def __init__(self, mode, recording_times=None, yubimoji_id=None, isPalmNormalised=False):
        self.mode = mode
        self.yubimoji_id = yubimoji_id
        self.recording_times = recording_times
        self.isPalmNormalised = isPalmNormalised

        # ラベルの読み込み
        labelFilePath = 'setting/hand_keypoint_classifier_label.csv'
        with open(labelFilePath, encoding='utf-8-sig') as f:
            self.yubimoji_labels = csv.reader(f)
            self.yubimoji_labels = [row[0] for row in self.yubimoji_labels]

        # yubimoji_idの値確認
        if mode == 0:
            if self.yubimoji_id is None or not (0 <= self.yubimoji_id <= 88):
                raise ValueError(
                    "Invalid yubimoji_id. It must be between 0 and 88.")
            else:
                print("yubimoji_id:", self.yubimoji_labels[self.yubimoji_id])

        self.feed_frames = FeedFrames(self.isPalmNormalised)
        self.calc = Calculate()
        self.write_csv = WriteCSV()

        self.setup()

    def setup(self):
        # Initialize working directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Load MediaPip
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        self.setup_camera()
        self.initialize_buffers()
        self.initialize_static_variables()
        self.initialize_variables()

    def setup_camera(self):
        self.video_capture = cv2.VideoCapture(0)
        self.frame_width, self.frame_height = 640, 360
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

    def initialize_buffers(self):
        self.landmarks_buffer = []
        self.lm_normalised_buffer = []
        self.results_buffer = []
        self.processingNo_buffer = []

    def initialize_static_variables(self):
        # 変数の初期化
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.buffer_size = 15  # int(self.fps * buffer_duration)

        self.frameCount = 0

        self.recordsCnt = 0 if self.mode == 0 and self.yubimoji_id is not None else None
        self.starttime = datetime.datetime.now() if self.mode == 0 else None

    def initialize_variables(self):
        # 変数の初期化

        self.noInputCount = 0
        self.InputCount = 0
        self.frameCount_buffer = -1
        self.noMovementCnt = 0

    def run(self):
        # Main loop
        if self.video_capture.isOpened():
            while True:
                if self.process_frame():
                    break

        # Release resources
        self.release_resources()

    def process_frame(self):
        ret, self.frame = self.video_capture.read()
        if not ret:
            return True

        self.frame = cv2.flip(self.frame, 1)

        # キー入力(ESC:プログラム終了)
        key = cv2.waitKey(1)
        if key == 27:
            return True

        # 録画モードの場合、録画回数カウント
        if self.mode == 0:
            sec_dif = datetime.datetime.now() - self.starttime
            sec_dif = sec_dif.total_seconds()

            if sec_dif > 3:
                time.sleep(1)
                self.starttime = datetime.datetime.now()
                self.recordsCnt += 1

            if self.recordsCnt >= self.recording_times:
                return True

        # 判定モードの場合、結果表示領域用にframe内の指定範囲を塗りつぶし
        if self.mode == 1:
            cv2.rectangle(self.frame, (550, 0), (
                int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ), (255, 255, 255), -1)

        # FPSの表示
            cv2.putText(self.frame, str(int(self.fps)), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # 検出処理の実行
        results = self.hands.process(
            cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

        # レイヤー描画用のインスタンス生成
        self.frame_overlay = FrameOverlay(self.frame)

        if results.multi_hand_landmarks:
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                self.process_landmarks(landmarks, handedness)

        else:
            # 入力がない場合
            self.handle_no_input()

        # 現在保持しているのフレームの番号を表示
        self.frameCount += 1

        # 画像の表示
        # cv2.imshow("MediaPipe Hands", self.frame)

        return False

    def process_landmarks(self, landmarks, handedness):

        # 手の左右判定 / 今回の副テーマ研究では左手の対応はしない
        if handedness.classification[0].label == 'Right':
            self.handle_right_hand(landmarks)
        else:
            self.handle_no_input()

    def handle_right_hand(self, landmarks):
        # 右手の処理

        # カウントなくても良い、確認用
        self.noInputCount = 0
        self.InputCount += 1

        if self.mode == 0:
            # 録画モードの処理
            self.handle_recording_mode(landmarks)
        elif self.mode == 1:
            # 判定モードの処理
            self.handle_prediction_mode(landmarks)

    def handle_recording_mode(self, landmarks):
        # 録画モードの処理

        # 正規化 ##################################################################
        # Landmarksのデータ型を表示
        landmark_list = self.calc.convert_to_pixel_coords(
            self.frame_width, self.frame_height, landmarks)

        # 相対座標に変換
        # 正規化は実施せずにそのままの座標を取得 / LSTMに通す前の事後処理にて正規化を都度実施
        relative_landmark_list = self.calc.convert_to_relative_coordinates(
            landmark_list)

        # 正規化は実施せずにそのままの座標を取得 / LSTMに通す前の事後処理にて正規化を都度実施

        # 画面表示 ################################################################
        # ランドマーク間の線を表示
        self.frame_overlay.lmLines(landmarks)

        # 文字表示
        # yubimoji = self.yubimoji_labels[self.yubimoji_id]
        # self.frame = self.frame_overlay.jpn_text(yubimoji)

        # 書き出し ################################################################
        self.write_csv.run(relative_landmark_list,
                           self.yubimoji_id, self.starttime)

    def handle_prediction_mode(self, landmarks):
        # 判定モードの処理

        # 文字判定後バッファがクリアされ、その後指定時間数 (15フレーム分) 待機し、それからバッファを貯めていく
        '''#if self.last_buffer_processed:

            # 1秒以上差が出てきたら以下を実行
            if ..... < 1:
                return
                self.last_buffer_processed_time = datetime.datetime.now()'''

        # 正規化:　Feed時ではなく、データ取得時に実施する
        # Landmarksのデータ型を表示
        landmark_list = self.calc.convert_to_pixel_coords(
            self.frame_width, self.frame_height, landmarks)

        # 相対座標に変換
        relative_landmark_list = self.calc.convert_to_relative_coordinates(
            landmark_list)

        # 確認用
        self.processingNo_buffer.append(self.frameCount)

        self.landmarks_buffer.append(relative_landmark_list)

        # 正規化後バッファが一定数に達したら、判定処理を実行
        if len(self.landmarks_buffer) > self.buffer_size*2:

            ####################################
            # 動作検出
            self.handle_movement_detection()

            # x軸、y軸の動作開始・停止をそれぞれ確認している場合
            if self.fn_flat_movement_detected is not None and self.fn_flat_movement_stopped is not None:

                frame_until = len(self.landmarks_buffer) - \
                    self.fn_flat_movement_stopped + 2
                landmarks_list_copy = self.landmarks_buffer[
                    self.fn_flat_movement_detected+1:frame_until]

                # バッファサイズ調整
                buffer_to_process = self.adjust_frame(landmarks_list_copy)
                print('dynamic-flat')

            # z軸の動作開始・停止をそれぞれ確認している場合 → 両変数間のフレームを動作として判定処理を流す
            elif self.fn_depth_movement_detected is not None and self.fn_depth_movement_stopped is not None:

                landmarks_list_copy = self.landmarks_buffer[
                    self.fn_depth_movement_detected:self.fn_depth_movement_stopped+1]

                # バッファサイズ調整
                buffer_to_process = self.adjust_frame(landmarks_list_copy)
                print('dynamic-depth')

            # 下記全ての変数がNoneの場合 → 静的指文字として判定処理を流す
            elif self.fn_flat_movement_detected is None and \
                    self.fn_flat_movement_stopped is None and \
                    self.fn_depth_movement_detected is None and \
                    self.fn_depth_movement_stopped is None:
                buffer_to_process = self.landmarks_buffer[:self.buffer_size]
                print('static')

            # 指定フレーム数内で動作開始、停止のどちらかしか検知できなかった場合
            else:
                print('error')
                return
            ####################################

            # サイズ調整後のバッファがNone、もしくはbuffer_size未満の場合
            if buffer_to_process is None or len(buffer_to_process) < self.buffer_size:
                self.initialize_buffers()
                return

            print('buffer_to_process:', len(buffer_to_process))
            print('='*20)

            # 判定処理に流す
            yubimoji, confidence = self.feed_frames.feed(buffer_to_process)
            '''
            self.results_buffer.append(
                [self.yubimoji_labels[yubimoji], confidence])

            # 画面表示
            self.frame = self.frame_overlay.results(self.results_buffer[-30:])

            # ランドマーク間の線を表示
            lm_latest = self.landmarks_buffer[-1]
            self.frame_overlay.lmLines(lm_latest)

            # 確認用
            print('\t', self.frameCount, self.yubimoji_labels[yubimoji],
                  '{:.2f}%'.format(confidence * 100))'''

            # 判定流したあとはバッファをクリア
            self.initialize_buffers()

    def handle_movement_detection(self):
        # 動作の検出

        landmarks_list = self.landmarks_buffer.copy()

        self.fn_flat_movement_detected = None
        self.fn_flat_movement_stopped = None
        self.fn_depth_movement_detected = None
        self.fn_depth_movement_stopped = None

        # 初期動作が収束したフレーム番号を検出 / 流したフレームで初期動作があった場合に収束フレームを検出
        fn_stabilised = self.calc.compare_frame_movement(
            landmarks_list, is_stop_detection=True)

        if fn_stabilised is not None:

            # 初期動作が収束以降のデータを取得
            landmarks_list = landmarks_list[fn_stabilised:]

            # 動作の開始を検出 / x-axis & y-axis
            # 過去5フレーム間の変化を検知
            # 検知した結果、動作が認められた場合その番号をfn_flat_movement_detectedに
            self.fn_flat_movement_detected = self.calc.compare_frame_movement(
                landmarks_list)

            print('-'*20)
            # 動作の停止を検出 / x-axis & y-axis
            # 同じ関数を使って、逆順にした過去5フレーム間の変化を検知
            # →後ろからフレームを走査し、動作停止フレームを検出、その番号をfn_flat_movement_stoppedに
            self.fn_flat_movement_stopped = self.calc.compare_frame_movement(
                landmarks_list[::-1])

            # 動作の開始・停止を検出 / z-axis
            # 過去5フレーム間の変化を検知
            self.fn_depth_movement_detected, self.fn_depth_movement_stopped = self.calc.compare_palm_length(
                landmarks_list)

        if self.fn_flat_movement_stopped != None and self.fn_flat_movement_detected != None:
            print('fn_stabilised:', fn_stabilised)
            print('fn_flat_movement_detected:', self.fn_flat_movement_detected)
            print('fn_flat_movement_stopped:', len(
                landmarks_list) - int(self.fn_flat_movement_stopped))

        if self.fn_depth_movement_detected != None and self.fn_depth_movement_stopped != None:
            print('fn_depth_movement_detected:',
                  self.fn_depth_movement_detected)
            print('fn_depth_movement_stopped:', self.fn_depth_movement_stopped)

    def adjust_frame(self, landmarks_list):

        if 1 < len(landmarks_list) < self.buffer_size:
            # フレーム数が足りない場合は、線形補間によるフレーム補填処理へ
            return self.calc.interpolate_frame(landmarks_list, self.buffer_size)

        elif len(landmarks_list) == self.buffer_size:
            # フレーム数がピッタリ設定数の場合は、そのまま
            return landmarks_list

        else:
            # フレームが1個以下、もしくはフレーム数が多い場合は無視
            return None

    def handle_no_input(self):
        # 入力がない場合の処理

        clearance_threshold = 75

        self.noInputCount += 1

        if self.noInputCount > clearance_threshold:
            self.initialize_buffers()
            self.initialize_variables()
            print(f'{self.frameCount}: Buffers cleared')

    def release_resources(self):
        # Release resources
        self.video_capture.release()
        self.hands.close()
        cv2.destroyAllWindows()


class Calculate:
    def __init__(self):
        pass

    def palm_length(self, landmarks):
        # 手掌長 (手首と中指第一関節間の距離) の取得 (Priyaら (2023))

        # 中指第一関節 - 手首の各軸の距離を計算
        distance_x = -landmarks[16]  # 0 - landmarks[16]
        distance_y = -landmarks[17]  # 0 - landmarks[17]

        # 中指第一関節 - 手首の距離を計算
        palm_length = math.sqrt(distance_x ** 2 + distance_y ** 2)

        return palm_length

    def convert_to_pixel_coords(self, frame_width, frame_height, landmarks, ignore_z=True):
        # ピクセル座標に変換

        landmark_list = []

        # 画面上のランドマークの位置を算出
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * frame_width), frame_width - 1)
            landmark_y = min(int(landmark.y * frame_height), frame_height - 1)

            if ignore_z:
                landmark_list.append([landmark_x, landmark_y])
            else:
                landmark_z = landmark.z
                landmark_list.append([landmark_x, landmark_y, landmark_z])

        return landmark_list

    def convert_to_relative_coordinates(self, landmark_list):
        # 相対座標に変換
        base_x, base_y = landmark_list[0][0], landmark_list[0][1]
        relative_landmark_list = [[x - base_x, y - base_y]
                                  for x, y in landmark_list]

        # 1次元リストに変換
        relative_landmark_list = list(
            itertools.chain.from_iterable(relative_landmark_list))

        # 最初の二行は0なので削除
        del relative_landmark_list[0:2]

        return relative_landmark_list

    def normalise(self, landmark_list, palm_length=None):
        # 手掌長で正規化

        norm_factor = max(
            abs(n) for n in landmark_list) if palm_length == None else palm_length

        # 正規化処理
        normalized_landmark_list = [n / norm_factor for n in landmark_list]
        return normalized_landmark_list

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

    def compare_frame_movement(self, landmarks_list, frame_focus=5, movement_threshold=2.0, is_stop_detection=False):

        for i in range(len(landmarks_list) - 1):

            nomovement_cnt = 0

            if i < frame_focus - 1:
                continue

            # 過去5フレーム分のランドマーク間の距離を計算
            for frame_ctrl in range(frame_focus):

                distance = self.detect_movement(
                    landmarks_list[i - frame_ctrl], landmarks_list[i + 1 - frame_ctrl])

                print(i - frame_ctrl, i + 1 - frame_ctrl, distance)

                if distance <= movement_threshold:
                    nomovement_cnt += 1

            print(i, nomovement_cnt)

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

            # detected_marker = ''

            if i < frame_focus:
                continue

            palm_length_cur = self.palm_length(landmarks_list[i])
            palm_length_comp = self.palm_length(
                landmarks_list[i - frame_focus])

            palm_length_dif = (palm_length_cur - palm_length_comp) ** 2

            if palm_length_dif >= dif_threshold:
                # detected_marker = '***'

                # 初回動作検知時、もしくは連続動作検知時の場合は、フレーム番号を更新
                if depth_change_detected_num == None:
                    depth_change_detected_num = i - frame_focus
                    depth_change_stopped_num = i

                if depth_change_detected_num != None and depth_change_stopped_num != None:
                    depth_change_stopped_num = i

            '''if self.is_check_mode:
                print('\t', i, palm_length_cur)
                print('\t', i - frame_focus, palm_length_comp)
                print('\t', palm_length_dif, detected_marker)
                print('='*20)'''

        return depth_change_detected_num, depth_change_stopped_num

    def interpolate_frame(self, landmarks_list, target_frames):

        new_landmark_array = np.array(landmarks_list)

        # フレーム数とデータ点の数を取得
        num_frames, num_points = new_landmark_array.shape

        # 新しいフレームを格納する配列
        frame_interpolated = np.zeros((target_frames, num_points))

        # interpolate
        for i in range(num_points):
            frame_interpolated[:, i] = np.interp(
                np.linspace(0, num_frames - 1, target_frames),
                np.arange(num_frames),
                new_landmark_array[:, i]
            )

        return frame_interpolated.tolist()


# フレーム描画用のクラス
class FrameOverlay:
    def __init__(self, frame):
        self.fontType = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'
        self.frame = frame
        self.frame_height, self.frame_width = self.frame.shape[:2]

        # imgをndarrayからPILに変換
        self.img_pil = Image.fromarray(self.frame)

        # drawインスタンス生成
        self.draw = ImageDraw.Draw(self.img_pil)

    def lmLines(self, landmarks):
        # 各ランドマークの座標は landmarks.landmark[0]~[20].x, .y, .z に格納されている

        # Landmark間の線
        landmark_line_ids = [
            (0, 1),     # 手首 - 親指第一関節
            (1, 5),     # 親指第一関節 - 人差し指第一関節
            (5, 9),     # 人差し指第一関節 - 中指第一関節
            (9, 13),    # 中指第一関節 - 薬指第一関節
            (13, 17),   # 薬指第一関節 - 小指第一関節
            (17, 0),    # 小指第一関節 - 手首
            (1, 2),     # 親指第一関節 - 親指第二関節
            (2, 3),     # 親指第二関節 - 親指第三関節
            (3, 4),     # 親指第三関節 - 親指先端
            (5, 6),     # 人差し指第一関節 - 人差し指第二関節
            (6, 7),     # 人差し指第二関節 - 人差し指第三関節
            (7, 8),     # 人差し指第三関節 - 人差し指先端
            (9, 10),    # 中指第一関節 - 中指第二関節
            (10, 11),   # 中指第二関節 - 中指第三関節
            (11, 12),   # 中指第三関節 - 中指先端
            (13, 14),   # 薬指第一関節 - 薬指第二関節
            (14, 15),   # 薬指第二関節 - 薬指第三関節
            (15, 16),   # 薬指第三関節 - 薬指先端
            (17, 18),   # 小指第一関節 - 小指第二関節
            (18, 19),   # 小指第二関節 - 小指第三関節
            (19, 20),   # 小指第三関節 - 小指先端
            (0, 9),     # 中指第一関節 - 手首
        ]

        # landmarkの繋がりをlineで表示
        for line_id in landmark_line_ids:
            # 1点目座標取得
            lm = landmarks.landmark[line_id[0]]
            lm_pos1 = (int(lm.x * self.frame_width),
                       int(lm.y * self.frame_height))
            # 2点目座標取得
            lm = landmarks.landmark[line_id[1]]
            lm_pos2 = (int(lm.x * self.frame_width),
                       int(lm.y * self.frame_height))
            # line描画
            cv2.line(self.frame, lm_pos1, lm_pos2, (128, 0, 0), 1)

        # landmarkをcircleで表示
        z_list = [lm.z for lm in landmarks.landmark]
        z_min = min(z_list)
        z_max = max(z_list)
        for lm in landmarks.landmark:
            lm_pos = (int(lm.x * self.frame_width),
                      int(lm.y * self.frame_height))
            lm_z = int((lm.z - z_min) / (z_max - z_min) * 255)
            cv2.circle(self.frame, lm_pos, 3, (255, lm_z, lm_z), -1)

    def palm_length(self, palm_length):
        cv2.putText(self.frame, str(palm_length), (200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (64, 0, 0), 1, cv2.LINE_AA)

    def jpn_text(self, text):

        font = ImageFont.truetype(self.fontType, size=20)

        # テキスト描画
        self.draw.text(
            xy=(20, 20),
            text=text,
            font=font,
            fill=(0, 0, 0)
        )

        # PILからndarrayに変換して返す
        return np.array(self.img_pil)

    def results(self, results_list):

        font = ImageFont.truetype(self.fontType, size=12)

        row_h = 18
        confidence_threshold = 0.5

        results_list = results_list[::-1]

        for i, (yubimoji, confidence) in enumerate(results_list):
            y_axis = 10 + i * row_h
            if y_axis + row_h > self.frame_height:
                break

            if confidence > confidence_threshold:

                result_txt = yubimoji + '\t{:.2f}%'.format(confidence * 100)

            else:
                'N/A'

            self.draw.text((560, y_axis), result_txt, font=font, fill=(
                0, 0, 0) if confidence > confidence_threshold else (128, 128, 128))

        self.frame = np.array(self.img_pil)


# recognition
class FeedFrames:
    def __init__(self, isPalmNormalised):

        self.isPalmNormalised = isPalmNormalised

        # 予測モデルのロード / あんまり分ける意味ないかも
        if self.isPalmNormalised:
            tflitePath = "palm_model/keypoint_classifier.tflite"
            self.keypoint_classifier = KeyPointClassifierwithPalm(tflitePath)
        else:
            tflitePath = "model/keypoint_classifier.tflite"
            self.keypoint_classifier = KeyPointClassifier(tflitePath)

        self.calc = Calculate()

    def feed(self, landmarks_buffer):

        lm_normalised_buffer = []

        # バッファ内の各フレームに対する処理
        for landmarks in landmarks_buffer:

            # 正規化
            if self.isPalmNormalised:
                # 手掌長の取得
                palm_length = self.calc.palm_length(landmarks)
                lm_normalised = self.calc.normalise(landmarks, palm_length)

            else:
                lm_normalised = self.calc.normalise(landmarks)

            # 正規化後のランドマークをバッファごとに保管
            lm_normalised_buffer.append(lm_normalised)

        # 文字の予測
        lm_list = np.array(lm_normalised_buffer, dtype=np.float32)
        lm_list = np.expand_dims(lm_list, axis=0)  # (1, 30, 40)

        yubimoji_id, confidence = self.keypoint_classifier(lm_list)

        # 結果の表示
        confidence_threshold = 0.5

        if confidence > confidence_threshold:
            return yubimoji_id, confidence

        else:
            no_yubimoji = 76  # 判定不可
            low_confidence = 0
            return no_yubimoji, low_confidence


# write csv
class WriteCSV:
    def __init__(self):
        self.distance_from_camera = 70
        pass

    def run(self, landmark_list, yubimoji_id=None, starttime=None, csv_path=None):
        # csv保存

        starttime = starttime.strftime('%Y%m%d%H%M%S') if starttime != None else datetime.datetime.now(
        ).strftime('%Y%m%d%H%M%S')  # 現在時刻の取得

        if csv_path == None:
            csv_path = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/point_history/0_point_history/point_history_{0}cm_{1}_{2}.csv'.format(
                str(self.distance_from_camera), str(yubimoji_id).zfill(2), starttime)

        with open(csv_path, 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerow([yubimoji_id, *landmark_list])


def main(mode, yubimoji_id=None, recording_times=None, isPalmNormalised=False):
    # インスタンスの生成と実行
    try:

        if mode == 0:
            # 録画モード
            hand_recognition = HandGestureRecognition(
                mode, yubimoji_id=yubimoji_id, recording_times=recording_times)

        elif mode == 1:
            # hand_recognition = HandGestureRecognition(mode)

            # 掌長で正規化したモデル
            hand_recognition = HandGestureRecognition(
                mode=1, isPalmNormalised=False)

        hand_recognition.run()

    except ValueError as e:
        print(f"{e}")


if __name__ == "__main__":
    main(1)

    '''
    # 録画モード
    times = 1
    start_id = 71
    end_id = start_id + 1

    for yubimoji_id in range(start_id, end_id):

        if yubimoji_id in [24, 34, 39] or yubimoji_id in range(44, 76):
            # 動的指文字

            for i in range(times):
                main(0, yubimoji_id=yubimoji_id, recording_times=1)

        else:
            main(0, yubimoji_id=yubimoji_id, recording_times=10)

        print("------------------")
        time.sleep(10)
    '''

'''
0: あ	1: い	2: う	3: え	4: お   5: か	6: き	7: く	8: け	9: こ	
10: さ	11: し	12: す	13: せ	14: そ  15: た	16: ち	17: つ	18: て	19: と	
20: な	21: に	22: ぬ	23: ね	24: の  25: は	26: ひ	27: ふ	28: へ	29: ほ	
30: ま	31: み	32: む	33: め	34: も  35: や	36: ゆ	37: よ	
38: ら	39: り	40: る	41: れ	42: ろ	43: わ	44: を	45: ん	
46: が	47: ぎ	48: ぐ	49: げ	50: ご	51: ざ	52: じ	53: ず	54: ぜ	55: ぞ
56: だ	57: ぢ	58: づ	59: で	60: ど	61: ば	62: び	63: ぶ	64: べ	65: ぼ
66: ぱ	67: ぴ	68: ぷ	69: ぺ	70: ぽ  71: っ  72: ゃ  73: ゅ  74 :ょ  75:ー
'''
