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

# --- 天気予報 API（livedoor 天気互換）の設定 ---
# 地域コードのマッピング (主要都市の例)
# 完全なリストは https://weather.tsukumijima.net/primary_area.xml で確認できます
CITY_CODES = {
    "選択してください": "", # 初期値
    "東京": "130010",
    "大阪": "270000",
    "名古屋": "230000",
    "福岡": "400010",
    "札幌": "016010",
    "仙台": "040010",
    "横浜": "140010",
    "京都": "260010",
    "広島": "340010",
    "那覇": "471000",
    "その他（地域コード入力）": "custom" # ユーザーが直接コードを入力するためのオプション
}

# 天気情報を取得する関数（livedoor 天気互換 API対応）
def get_weather_by_city_code(city_code):
    """指定された地域コードの天気情報をlivedoor 天気互換 APIから取得します。"""
    WEATHER_API_URL_LIVEDOOR = f"https://weather.tsukumijima.net/api/forecast/city/{city_code}"
    try:
        response = requests.get(WEATHER_API_URL_LIVEDOOR)
        response.raise_for_status() # HTTPエラーがあれば例外を発生させる
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        st.error(f"天気情報の取得中にエラーが発生しました: {e}")
        return None

# --- 洗濯物乾きやすさ判断ロジック関数 (livedoor APIのデータに合わせて調整) ---
def get_laundry_drying_recommendation_livedoor(telop, max_temp, min_temp, chance_of_rain_periods):
    """
    livedoor 天気互換 APIのデータに基づいて洗濯物の乾きやすさとおすすめの乾燥方法を判断する。
    Args:
        telop (str): 天気概況 (例: "晴れ", "曇り", "雨")
        max_temp (float or None): 最高気温
        min_temp (float or None): 最低気温
        chance_of_rain_periods (list): 時間帯ごとの降水確率リスト (例: [10, 20, 30, 40])
    Returns:
        dict: 乾きやすさのステータスとおすすめメッセージ
    """
    drying_status = "不明"
    recommendation = "天気情報が不足しているか、判断できません。"

    # まず雨や雪が降る予報、または降水確率が高いかをチェック
    # telopに「雨」や「雪」が含まれる場合
    if "雨" in telop or "雪" in telop:
        drying_status = "室内干し推奨"
        recommendation = "雨や雪が降る予報です。洗濯物は屋外では乾きません。室内干しにするか、乾燥機の利用を検討してください。"
    # いずれかの時間帯で降水確率が50%を超える場合
    elif any(cor > 50 for cor in chance_of_rain_periods if cor is not None):
        drying_status = "室内干し推奨"
        recommendation = "降水確率が高い時間帯があります。急な雨に注意し、室内干しを検討してください。"
    # 最高気温が極端に低い場合 (乾燥しにくい)
    elif max_temp is not None and max_temp < 5:
        drying_status = "非常に乾きにくい"
        recommendation = "気温が低すぎます。外干しは非常に乾きにくいでしょう。乾燥機や浴室乾燥の利用をおすすめします。"
    else:
        # 天気概況と最高気温を基に判断 (livedoor APIでは湿度や風速のリアルタイム値がないため簡略化)
        if "晴れ" in telop:
            if max_temp is not None and max_temp >= 25:
                drying_status = "非常に乾きやすい"
                recommendation = "最高の洗濯日和です！太陽が洗濯物をあっという間に乾かしてくれます。"
            elif max_temp is not None and max_temp >= 20:
                drying_status = "乾きやすい"
                recommendation = "外干しに適しています。気持ちよく乾きますよ。"
            else:
                drying_status = "普通"
                recommendation = "晴れですが、気温によっては乾きに時間がかかるかもしれません。風通しを良くしましょう。"
        elif "曇り" in telop or "くもり" in telop:
            if max_temp is not None and max_temp >= 15:
                drying_status = "普通"
                recommendation = "曇り空ですが、外干しは可能です。厚手のものは乾くのに時間がかかるかも。"
            else:
                drying_status = "乾きにくい"
                recommendation = "曇りで気温も低めです。乾きにくい一日なので、室内干しや乾燥機の検討を。"
        else: # その他の天気 (例: 晴れ時々曇りなど)
            drying_status = "普通"
            recommendation = "今日の天気は外干し可能ですが、状況によっては乾きにくいかもしれません。天気予報をよく確認しましょう。"

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
    
    results = model(img_np, conf=0.2, verbose=False)

    detected_marks = []
    washing_instructions = []
    show_weather_section_based_on_detection = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            class_name = model.names[cls]
            
            if conf > 0.2:
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

        # 修正: display_option を display_weather_option に変更
        if display_weather_option == "はい":
            st.subheader("お洗濯の前に天気予報を確認しましょう！")
            
            # --- 都市名入力部分をlivedoor API向けに修正 ---
            selected_city_name = st.selectbox(
                "天気予報を表示したい都市を選択してください:",
                list(CITY_CODES.keys()),
                key="city_select"
            )

            city_code = ""
            if selected_city_name == "その他（地域コード入力）":
                custom_city_code = st.text_input(
                    "地域コードを直接入力してください (例: 東京:130010):",
                    key="custom_city_code_input"
                )
                if custom_city_code:
                    city_code = custom_city_code
            elif selected_city_name:
                city_code = CITY_CODES[selected_city_name]
            # --- 修正ここまで ---
            
            if st.button("天気予報を取得", key="get_weather_button") and city_code:
                with st.spinner("天気情報を取得中..."):
                    # livedoor APIを使用
                    weather_data = get_weather_by_city_code(city_code)
                    
                    if weather_data:
                        # 修正: 'forecasts' キーの存在を安全に確認
                        forecasts = weather_data.get('forecasts')
                        
                        if not forecasts:
                            st.error("天気予報データが見つかりませんでした。選択した地域コードが正しいか、またはAPIのレスポンスが予期せぬ形式です。")
                            return # データがない場合はここで処理を中断
                            
                        today_forecast = forecasts[0] # 今日の予報
                        
                        display_location = weather_data.get('location', {}).get('area', '指定された地域')
                        st.success(f"{display_location}の今日の天気:")
                        
                        st.write(f"**天気:** {today_forecast['telop']}")
                        
                        # 最高気温と最低気温は 'temperature' キーの下にある
                        max_temp = today_forecast['temperature'].get('max', {}).get('celsius')
                        min_temp = today_forecast['temperature'].get('min', {}).get('celsius')
                        
                        if max_temp:
                            st.write(f"**最高気温:** {max_temp} °C")
                        if min_temp:
                            st.write(f"**最低気温:** {min_temp} °C")
                        
                        # 降水確率は時間帯ごとに取得
                        chance_of_rain_periods = []
                        st.write("**降水確率:**")
                        for period, value in today_forecast['chanceOfRain'].items():
                            if value: # 値が存在する場合のみ表示
                                # '%%' を取り除いて表示
                                cleaned_value = value.replace('%%', '')
                                st.write(f"　{period.replace('T', '').replace('_', ':00-')}: {cleaned_value}%")
                                try:
                                    # '%%' を取り除いてからfloatに変換
                                    chance_of_rain_periods.append(float(cleaned_value))
                                except ValueError:
                                    chance_of_rain_periods.append(0) # 変換できない場合は0

                        # --- 洗濯物乾きやすさインジケーターとおすすめ乾燥方法を表示 (livedoor API向け) ---
                        # livedoor APIには湿度や風速の直接的な現在の情報がないため、telopと気温、降水確率で判断
                        drying_info = get_laundry_drying_recommendation_livedoor(
                            today_forecast['telop'],
                            float(max_temp) if max_temp else None,
                            float(min_temp) if min_temp else None,
                            chance_of_rain_periods
                        )
                        
                        st.markdown("#### 洗濯物の乾きやすさ:")
                        if drying_info["drying_status"] == "非常に乾きやすい":
                            st.success(f"**{drying_info['drying_status']}**")
                        elif drying_info["drying_status"] == "乾きやすい":
                            st.success(f"**{drying_status['drying_status']}**")
                        elif drying_info["drying_status"] == "普通":
                            st.info(f"**{drying_info['drying_status']}**")
                        elif drying_info["drying_status"] == "乾きにくい":
                            st.warning(f"**{drying_info['drying_status']}**")
                        else: # 室内干し推奨
                            st.error(f"**{drying_info['drying_status']}**")
                        
                        st.write(f"**アドバイス:** {drying_info['recommendation']}")

                    else:
                        st.error("指定された地域コードの天気情報を取得できませんでした。地域コードが正しいか確認してください。")
            elif st.button("天気予報を取得", key="get_weather_button_no_city_code") and not city_code:
                st.warning("都市を選択するか、地域コードを入力してください。")
