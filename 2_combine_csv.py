##############################################################################
# 正規化していないlandmarksを後から正規化する際に使う
##############################################################################

# Standard Library
import glob
import os

# Third-Party Libraries
import pandas as pd


def main(buffer_size=15):

    # 入力元・出力先指定
    parent_dir = '/Users/yoshifumihanada/Documents/2_study/1_修士/3_副研究/'
    import_crt = '1_point_history_resized/*.csv'
    export_file = 'point_history/point_history_combined.csv'

    import_dir = parent_dir + import_crt
    export_file = parent_dir + export_file

    # 出力先のファイルが存在する場合、削除
    if os.path.exists(export_file):
        os.remove(export_file)

    # パスで指定したファイルの一覧をリスト形式で取得
    csv_files = glob.glob(import_dir)

    # csvファイルの中身を追加していくリストを用意
    data_list = []

    # 読み込むファイルのリストを走査
    for file in csv_files:

        # ファイルを読み込む
        csv_ = pd.read_csv(file, header=None)

        if len(csv_) == buffer_size:
            csv_ = csv_.tail(buffer_size)
            data_list.append(csv_)
        else:
            print('File size is not enough.')

    # リストを全て行方向に結合
    df = pd.concat(data_list, axis=0, ignore_index=True)

    # 保存
    df.to_csv(export_file, index=False, header=False)


main()
