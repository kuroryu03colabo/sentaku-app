import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import requests # requestsライブラリをインポート

# Streamlitアプリのタイトル
st.title("洗濯表示マーク判別と天気予報")

st.write("洗濯表示タグの画像をアップロードしてください。")

# YOLOモデルのロード
# モデルファイルのパスを適切に指定してください
try:
    model = YOLO("best_2.pt")
except Exception as e:
    st.error(f"モデルのロードに失敗しました。'best.pt'ファイルが正しいパスにあるか確認してください。エラー: {e}")
    st.stop()

# OpenWeatherMap APIキーの設定
# ここにあなたのOpenWeatherMap APIキーを貼り付けてください
OPENWEATHER_API_KEY = "58e24451bd8fb05efb390ac5ef3311be"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# 天気情報を取得する関数
def get_weather(city_name, api_key):
    """指定された都市の天気情報をOpenWeatherMap APIから取得します。"""
    params = {
        "q": city_name,
        "appid": api_key,
        "units": "metric", # 摂氏で取得
        "lang": "ja" # 日本語で取得
    }
    try:
        response = requests.get(WEATHER_API_URL, params=params)
        response.raise_for_status() # HTTPエラーがあれば例外を発生させる
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        st.error(f"天気情報の取得中にエラーが発生しました: {e}")
        return None

# アップロードされた画像の処理
uploaded_file = st.file_uploader("画像を選択してください...", type=["jpg", "jpeg", "png"])

# 天気予報を表示するかどうかのフラグ
show_weather_section_based_on_detection = False

if uploaded_file is not None:
    # 画像をPIL Imageとして開く
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_container_width=True)
    st.write("")
    st.write("判別中...")

    # PIL ImageをOpenCV形式（NumPy配列）に変換
    img_np = np.array(image)
    # PILはRGB、OpenCVはBGRなので変換が必要な場合があります
    if img_np.shape[2] == 3: # カラー画像の場合
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # YOLOv8による推論
    results = model(img_np)

    # 推論結果の表示
    detected_marks = []
    washing_instructions = [] # 洗濯指示を格納するリスト

    # 推論結果からバウンディングボックスとクラス情報を取得
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            class_name = model.names[cls] # クラス名をモデルから取得
            
            # 信頼度（confidence）を0.2以上のもののみ表示
            if conf > 0.2:
                detected_marks.append(f"• {class_name} (信頼度: {conf:.2f})")
                
                # 洗濯指示のロジックを追加
                if class_name == "HW_OK":
                    washing_instructions.append("手洗いコースを選んでください。")
                    show_weather_section_based_on_detection = True # 手洗いOKの場合も天気予報セクションを表示
                elif class_name == "LD_OK":
                    washing_instructions.append("洗濯機での洗濯ができます。")
                    show_weather_section_based_on_detection = True # 洗濯機OKの場合も天気予報セクションを表示
                elif class_name == "LD_NG":
                    washing_instructions.append("洗濯機で洗濯できません。")
                
                # バウンディングボックスを描画する準備 (オプション)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2) # 緑色の矩形
                cv2.putText(img_np, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)    
                
    if detected_marks:
        st.success("検出された洗濯表示マーク:")
        for mark in detected_marks:
            st.write(mark)
        
        st.subheader("洗濯に関する指示:")
        if washing_instructions:
            # 重複するメッセージを避けるためにsetを使用
            for instruction in sorted(list(set(washing_instructions))):
                st.info(instruction)
        else:
            st.info("洗濯に関する具体的な指示は検出されませんでした。")

        # 検出結果が描画された画像をStreamlitで表示 (オプション)
        st.subheader("検出結果（画像）")
        # BGRからRGBに変換し直して表示
        img_with_detections = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        st.image(img_with_detections, caption="検出結果", use_column_width=True)

    else:
        st.info("洗濯表示マークは検出されませんでした。")

# 洗濯OKまたは手洗いOKが検出された場合に、天気予報セクションを表示するか選択する
if show_weather_section_based_on_detection:
    st.markdown("---") # 区切り線
    
    # ユーザーに天気予報を表示するかどうかを選択してもらう
    display_weather_option = st.radio(
        "お洗濯の前に天気予報を確認しますか？",
        ("はい", "いいえ"),
        key="weather_display_choice",
        horizontal=True # 横並びにする
    )

    if display_weather_option == "はい":
        st.subheader("お洗濯の前に天気予報を確認しましょう！")
        
        # ユーザーに地域名を入力してもらう
        city_name = st.text_input("天気予報を表示したい地域名を入力してください (例: Tokyo):", key="city_input")
        
        if st.button("天気予報を取得", key="get_weather_button") and city_name:
            if OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY":
                st.warning("OpenWeatherMap APIキーを設定してください。")
            else:
                with st.spinner("天気情報を取得中..."):
                    weather_data = get_weather(city_name, OPENWEATHER_API_KEY)
                    
                    if weather_data:
                        st.success(f"{city_name}の現在の天気:")
                        st.write(f"**天気:** {weather_data['weather'][0]['description']}")
                        st.write(f"**気温:** {weather_data['main']['temp']} °C")
                        st.write(f"**最高気温:** {weather_data['main']['temp_max']} °C")
                        st.write(f"**最低気温:** {weather_data['main']['temp_min']} °C")
                        st.write(f"**湿度:** {weather_data['main']['humidity']} %")
                        st.write(f"**風速:** {weather_data['wind']['speed']} m/s")
                        
                        # 雨の可能性を示唆する簡単なロジック
                        if 'rain' in weather_data and weather_data['rain'].get('1h', 0) > 0:
                            st.warning(f"**注意:** 過去1時間に {weather_data['rain']['1h']} mm の降水がありました。")
                        elif 'clouds' in weather_data and weather_data['clouds']['all'] > 50:
                            st.info("雲が多いですが、現在のところ降水はありません。")
                        else:
                            st.info("洗濯日和です！")
                    else:
                        st.error("指定された地域の天気情報を取得できませんでした。地域名が正しいか確認してください。")
        elif st.button("天気予報を取得", key="get_weather_button_no_city") and not city_name:
            st.warning("地域名を入力してください。")

