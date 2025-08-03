import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import requests # requestsライブラリをインポート
import os # 環境変数を扱うためにインポート

# Streamlitアプリのタイトル
st.markdown("### 洗濯表示マーク判別アプリと天気予報") # 文字サイズを小さくして一行に収める

st.write("洗濯表示タグの画像をアップロードしてください。")

# YOLOモデルのロード
# モデルファイルのパスを適切に指定してください（例: best_5.pt）
MODEL_PATH = "best_5.pt"
try:
    model = YOLO(MODEL_PATH)
    # st.success(f"YOLOv8モデルをロードしました: {MODEL_PATH}") # この行は非表示に
except Exception as e:
    st.error(f"モデルのロードに失敗しました。'{MODEL_PATH}'ファイルが正しいパスにあるか確認してください。エラー: {e}")
    st.stop() # モデルがロードできない場合はアプリを停止

# --- Open-Meteo API の設定 ---
OPENMETEO_API_URL = "https://api.open-meteo.com/v1/forecast"

# --- 都市と緯度経度のデータベース（Python辞書） ---
# 各都道府県の県庁所在地と主要都市の緯度経度
CITY_COORDINATES = {
    "選択してください": {"latitude": None, "longitude": None},
    # 北海道地方
    "札幌": {"latitude": 43.0621, "longitude": 141.3544},
    # 東北地方
    "青森": {"latitude": 40.8222, "longitude": 140.7474},
    "盛岡": {"latitude": 39.7027, "longitude": 141.1357},
    "仙台": {"latitude": 38.2682, "longitude": 140.8694},
    "秋田": {"latitude": 39.7186, "longitude": 140.1024},
    "山形": {"latitude": 38.2554, "longitude": 140.3633},
    "福島": {"latitude": 37.7502, "longitude": 140.4670},
    # 関東地方
    "水戸": {"latitude": 36.3708, "longitude": 140.4704},
    "宇都宮": {"latitude": 36.5551, "longitude": 139.8821},
    "前橋": {"latitude": 36.3912, "longitude": 139.0605},
    "さいたま": {"latitude": 35.8569, "longitude": 139.6489},
    "千葉": {"latitude": 35.6049, "longitude": 140.1233},
    "東京": {"latitude": 35.6895, "longitude": 139.6917},
    "横浜": {"latitude": 35.4437, "longitude": 139.6380},
    # 中部地方
    "新潟": {"latitude": 37.9023, "longitude": 139.0232},
    "富山": {"latitude": 36.6953, "longitude": 137.2114},
    "金沢": {"latitude": 36.5612, "longitude": 136.6562},
    "福井": {"latitude": 36.0652, "longitude": 136.2217},
    "甲府": {"latitude": 35.6639, "longitude": 138.5683},
    "長野": {"latitude": 36.6513, "longitude": 138.1824},
    "岐阜": {"latitude": 35.3970, "longitude": 136.7222},
    "静岡": {"latitude": 34.9769, "longitude": 138.3838},
    "名古屋": {"latitude": 35.1815, "longitude": 136.9066},
    # 近畿地方
    "津": {"latitude": 34.7360, "longitude": 136.5085},
    "大津": {"latitude": 35.0063, "longitude": 135.8685},
    "京都": {"latitude": 35.0116, "longitude": 135.7681},
    "大阪": {"latitude": 34.6937, "longitude": 135.5023},
    "神戸": {"latitude": 34.6901, "longitude": 135.1955},
    "奈良": {"latitude": 34.6851, "longitude": 135.8048},
    "和歌山": {"latitude": 34.2260, "longitude": 135.1675},
    # 中国地方
    "鳥取": {"latitude": 35.5036, "longitude": 134.2370},
    "松江": {"latitude": 35.4723, "longitude": 133.0504},
    "岡山": {"latitude": 34.6617, "longitude": 133.9211},
    "広島": {"latitude": 34.3963, "longitude": 132.4596},
    "山口": {"latitude": 34.1857, "longitude": 131.4714},
    # 四国地方
    "徳島": {"latitude": 34.0658, "longitude": 134.5594},
    "高松": {"latitude": 34.3401, "longitude": 134.0467},
    "松山": {"latitude": 33.8417, "longitude": 132.7661},
    "高知": {"latitude": 33.5597, "longitude": 133.5311},
    # 九州・沖縄地方
    "福岡": {"latitude": 33.5904, "longitude": 130.4017},
    "佐賀": {"latitude": 33.2494, "longitude": 130.2988},
    "長崎": {"latitude": 32.7503, "longitude": 129.8776},
    "熊本": {"latitude": 32.7898, "longitude": 130.7417},
    "大分": {"latitude": 33.2382, "longitude": 131.6125},
    "宮崎": {"latitude": 31.9111, "longitude": 131.4239},
    "鹿児島": {"latitude": 31.5603, "longitude": 130.5580},
    "那覇": {"latitude": 26.2124, "longitude": 127.6809},
    # --- 追加された都市ここまで ---
    "その他（緯度経度入力）": {"latitude": None, "longitude": None} # ユーザーが直接入力するためのオプション
}


# 天気情報を取得する関数（緯度経度対応 - Open-Meteo API）
def get_weather_by_lat_lon(latitude, longitude):
    """指定された緯度経度の天気情報をOpen-Meteo APIから取得します。"""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,showers,snowfall,weather_code,wind_speed_10m",
        "timezone": "Asia/Tokyo",
        "forecast_days": 1 # 現在の天気のみが必要
    }
    try:
        response = requests.get(OPENMETEO_API_URL, params=params)
        response.raise_for_status() # HTTPエラーがあれば例外を発生させる
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        st.error(f"天気情報の取得中にエラーが発生しました: {e}")
        return None

# --- 洗濯物乾きやすさ判断ロジック関数 (Open-Meteo APIのデータに合わせて調整) ---
def get_laundry_drying_recommendation(temp, humidity, wind_speed, precipitation, weather_code):
    """
    Open-Meteo APIのデータに基づいて洗濯物の乾きやすさとおすすめの乾燥方法を判断する。
    Args:
        temp (float): 現在の気温
        humidity (float): 現在の湿度
        wind_speed (float): 現在の風速
        precipitation (float): 現在の降水量 (rain + showers + snowfall)
        weather_code (int): 天気コード (WMO Weather interpretation codes)
    Returns:
        dict: 乾きやすさのステータスとおすすめメッセージ
    """
    drying_status = "不明"
    recommendation = "天気情報が不足しているか、判断できません。"

    # WMO Weather interpretation codes (主要な天気コードを簡略化)
    # 0: Clear sky
    # 1, 2, 3: Mainly clear, partly cloudy, overcast
    # 45, 48: Fog
    # 51, 53, 55: Drizzle
    # 56, 57: Freezing Drizzle
    # 61, 63, 65: Rain
    # 66, 67: Freezing Rain
    # 71, 73, 75: Snow fall
    # 77: Snow grains
    # 80, 81, 82: Rain showers
    # 85, 86: Snow showers
    # 95: Thunderstorm
    # 96, 99: Thunderstorm with hail

    # まず降水があるか、または降水を示す天気コードかをチェック
    if precipitation > 0.1 or weather_code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99]:
        drying_status = "室内干し推奨"
        recommendation = "雨や雪が降っています。洗濯物は屋外では乾きません。室内干しにするか、乾燥機の利用を検討してください。"
    elif temp is not None and temp < 5: # 極端に寒い場合
        drying_status = "非常に乾きにくい"
        recommendation = "気温が低すぎます。外干しは非常に乾きにくいでしょう。乾燥機や浴室乾燥の利用をおすすめします。"
    else:
        # 気温、湿度、風速、天気コードを総合的に判断するためのスコア
        score = 0
        if temp is not None:
            if temp >= 25: score += 3
            elif temp >= 20: score += 2
            elif temp >= 15: score += 1

        if humidity is not None:
            if humidity <= 50: score += 3
            elif humidity <= 65: score += 2
            elif humidity <= 75: score += 1

        if wind_speed is not None:
            if wind_speed >= 3: score += 3
            elif wind_speed >= 2: score += 2
            elif wind_speed >= 1: score += 1

        if weather_code is not None:
            if weather_code in [0, 1]: # Clear sky, Mainly clear
                score += 2
            elif weather_code in [2, 3]: # Partly cloudy, Overcast
                score += 1

        if score >= 7:
            drying_status = "非常に乾きやすい"
            recommendation = "最高の洗濯日和です！太陽と風が洗濯物をあっという間に乾かしてくれます。"
        elif score >= 5:
            drying_status = "乾きやすい"
            recommendation = "外干しに適しています。気持ちよく乾きますよ。"
        elif score >= 3:
            drying_status = "普通"
            recommendation = "外干しは可能ですが、厚手のものは乾くのに時間がかかるかもしれません。風通しを良くしましょう。"
        else:
            drying_status = "乾きにくい"
            recommendation = "乾きにくい一日です。可能であれば、室内干しや乾燥機の利用を検討してください。"
            if humidity is not None and humidity > 80:
                recommendation += "特に湿度が高いので、除湿器の利用も効果的です。"
            elif wind_speed is not None and wind_speed < 1:
                recommendation += "風が弱いので、扇風機で空気を循環させると良いでしょう。"

    return {"drying_status": drying_status, "recommendation": recommendation}


# アップロードされた画像の処理
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("画像を選択してください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col2:
        st.image(image, caption="アップロードされた画像", use_container_width=True)
    
    st.write("")
    st.write("判別中...")

    img_np = np.array(image)
    if img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    results = model(img_np, conf=0.6, verbose=False)

    detected_marks = []
    washing_instructions = []
    show_weather_section_based_on_detection = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            class_name = model.names[cls]
            
            if conf > 0.6:
                detected_marks.append(f"• {class_name} (信頼度: {conf:.2f})")
                
                if class_name == "HW_OK":
                    washing_instructions.append("手洗いコースを選んでください。")
                    show_weather_section_based_on_detection = True
                elif class_name == "LD_OK":
                    washing_instructions.append("洗濯機での洗濯ができます。")
                    show_weather_section_based_on_detection = True
                elif class_name == "LD_NG":
                    if "洗濯機で洗濯できません。" not in washing_instructions:
                        washing_instructions.append("洗濯機で洗濯できません。")
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_np, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)     
                
    if detected_marks:
        col_instructions, col_image_results = st.columns(2)

        with col_instructions:
            st.subheader("洗濯に関する指示:")
            if washing_instructions:
                for instruction in sorted(list(set(washing_instructions))):
                    st.info(instruction)
            else:
                st.info("洗濯に関する具体的な指示は検出されませんでした。")

        with col_image_results:
            st.subheader("検出結果（画像）")
            img_with_detections = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            st.image(img_with_detections, caption="検出結果", use_container_width=True)

    else:
        st.info("洗濯表示マークは検出されませんでした。")

    st.subheader("検出されたマークと信頼度:")
    if detected_marks:
        for mark in detected_marks:
            st.write(mark)
    else:
        st.info("検出された洗濯表示マークはありませんでした。")
    
    if show_weather_section_based_on_detection:
        st.markdown("---")
        display_weather_option = st.radio(
            "お洗濯の前に天気予報を確認しますか？",
            ("はい", "いいえ"),
            key="weather_display_choice",
            horizontal=True
        )

        if display_weather_option == "はい":
            st.subheader("お洗濯の前に天気予報を確認しましょう！")
            
            # --- 都市選択と緯度経度自動入力部分 ---
            selected_city_name = st.selectbox(
                "天気予報を表示したい都市を選択してください:",
                list(CITY_COORDINATES.keys()),
                key="city_select"
            )

            # 選択された都市に基づいて緯度経度を自動設定
            if selected_city_name and selected_city_name != "その他（緯度経度入力）":
                latitude = CITY_COORDINATES[selected_city_name]["latitude"]
                longitude = CITY_COORDINATES[selected_city_name]["longitude"]
                # Streamlitのテキスト入力フィールドにデフォルト値を設定
                # ただし、st.text_inputはkeyが同じだと値を保持するため、
                # selected_city_nameが変わったときにのみ更新されるようにする
                if 'last_selected_city' not in st.session_state or st.session_state.last_selected_city != selected_city_name:
                    st.session_state.latitude_input = str(latitude) if latitude is not None else ""
                    st.session_state.longitude_input = str(longitude) if longitude is not None else ""
                    st.session_state.last_selected_city = selected_city_name
            
            # 緯度経度入力フィールド（自動入力または手動入力用）
            latitude_str = st.text_input(
                "緯度 (例: 35.6895)",
                value=st.session_state.get('latitude_input', ''), # セッションステートから値を取得
                key="latitude_input"
            )
            longitude_str = st.text_input(
                "経度 (例: 139.6917)",
                value=st.session_state.get('longitude_input', ''), # セッションステートから値を取得
                key="longitude_input"
            )

            latitude = None
            longitude = None
            if latitude_str:
                try:
                    latitude = float(latitude_str)
                except ValueError:
                    st.error("緯度は数値で入力してください。")
            if longitude_str:
                try:
                    longitude = float(longitude_str)
                except ValueError:
                    st.error("経度は数値で入力してください。")
            # --- 都市選択と緯度経度自動入力部分ここまで ---
            
            if st.button("天気予報を取得", key="get_weather_button") and latitude is not None and longitude is not None:
                with st.spinner("天気情報を取得中..."):
                    weather_data = get_weather_by_lat_lon(latitude, longitude)
                    
                    if weather_data and 'current' in weather_data:
                        current_weather = weather_data['current']
                        
                        # Open-Meteoの天気コードから日本語の天気概況を取得する簡易マッピング
                        # 詳細なマッピングはOpen-Meteoのドキュメントを参照
                        weather_code_map = {
                            0: "快晴", 1: "晴れ", 2: "一部曇り", 3: "曇り",
                            45: "霧", 48: "霧氷",
                            51: "霧雨", 53: "霧雨", 55: "激しい霧雨",
                            61: "小雨", 63: "雨", 65: "大雨",
                            80: "小雨のにわか雨", 81: "雨のにわか雨", 82: "激しい雨のにわか雨",
                            71: "小雪", 73: "雪", 75: "大雪", 85: "小雪のにわか雨", 86: "大雪のにわか雨",
                            95: "雷雨", 96: "ひょうを伴う雷雨", 99: "激しいひょうを伴う雷雨"
                        }
                        weather_description = weather_code_map.get(current_weather.get('weather_code'), "不明")

                        st.success(f"緯度: {latitude}, 経度: {longitude} の現在の天気:")
                        
                        st.write(f"**天気:** {weather_description}")
                        st.write(f"**気温:** {current_weather.get('temperature_2m')} °C")
                        st.write(f"**湿度:** {current_weather.get('relative_humidity_2m')} %")
                        st.write(f"**風速:** {current_weather.get('wind_speed_10m')} m/s")
                        
                        # 降水量はrain, showers, snowfallの合計
                        precipitation_sum = current_weather.get('precipitation', 0)
                        st.write(f"**降水量 (過去1時間):** {precipitation_sum} mm")

                        # --- 洗濯物乾きやすさインジケーターとおすすめ乾燥方法を表示 (Open-Meteo API向け) ---
                        drying_info = get_laundry_drying_recommendation(
                            current_weather.get('temperature_2m'),
                            current_weather.get('relative_humidity_2m'),
                            current_weather.get('wind_speed_10m'),
                            precipitation_sum,
                            current_weather.get('weather_code')
                        )
                        
                        st.markdown("#### 洗濯物の乾きやすさ:")
                        if drying_info["drying_status"] == "非常に乾きやすい":
                            st.success(f"**{drying_info['drying_status']}**")
                        elif drying_info["drying_status"] == "乾きやすい":
                            st.success(f"**{drying_info['drying_status']}**")
                        elif drying_info["drying_status"] == "普通":
                            st.info(f"**{drying_info['drying_status']}**")
                        elif drying_info["drying_status"] == "乾きにくい":
                            st.warning(f"**{drying_info['drying_status']}**")
                        else: # 室内干し推奨
                            st.error(f"**{drying_info['drying_status']}**")
                        
                        st.write(f"**アドバイス:** {drying_info['recommendation']}")

                    else:
                        st.error("指定された緯度経度の天気情報を取得できませんでした。緯度経度が正しいか確認してください。")
            elif st.button("天気予報を取得", key="get_weather_button_no_coords") and (latitude is None or longitude is None):
                st.warning("緯度と経度を入力してください。")
