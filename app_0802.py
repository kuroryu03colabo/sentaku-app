# -*- coding: utf-8 -*-
import streamlit as st
import io
import requests
from PIL import Image
import numpy as np
from ultralytics import YOLO
import geocoder # ユーザーの位置情報を取得するために使用
import os

# --- アプリケーション設定 ---
st.set_page_config(
    page_title="洗濯表示タグ識別アプリ",
    page_icon="👕",
    layout="wide"
)

# タイトル
st.title("👕 洗濯表示タグ識別アプリ")
st.markdown("洗濯表示タグの画像をアップロードするか、カメラで撮影して、洗濯方法と外干し/部屋干しのアドバイスを取得します。")

# --- モデルとAPIの設定 ---
# YOLOv8モデルをロードします。
# デプロイメントを考慮し、モデルファイル(best_5.pt)はスクリプトと同じディレクトリにあることを想定しています。
try:
    model_path = "best_5.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        st.error(f"モデルファイルが見つかりません。{model_path}をスクリプトと同じディレクトリに配置してください。")
        st.stop()
except Exception as e:
    st.error(f"モデルのロード中にエラーが発生しました: {e}")
    st.stop()

# APIキーのセキュリティに関する注意
# Open-Meteo APIはAPIキーを必要としませんが、他のサービスでAPIキーを使用する際は
# Streamlit Cloudのシークレット機能 `st.secrets` を使用することを強く推奨します。
# `secrets.toml` ファイルに以下のように記述します:
# [api_keys]
# open_meteo_dummy = "your_secret_key"
# st.secrets.api_keys.open_meteo_dummy でアクセスできます。

# --- UI要素 ---
# 画像入力方法の選択
input_method = st.sidebar.radio(
    "画像入力方法を選択してください",
    ("カメラで撮影", "画像をアップロード")
)

uploaded_file = None
if input_method == "カメラで撮影":
    st.markdown("カメラで洗濯表示タグを撮影してください。")
    uploaded_file = st.camera_input("カメラ起動")
else:
    st.markdown("洗濯表示タグの画像をアップロードしてください。")
    uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

# --- 天気情報から洗濯物の乾きやすさを判断するロジック ---
def determine_drying_conditions(temp, humidity, wind_speed, precipitation, weather_code):
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

# --- 処理の実行 ---
if uploaded_file is not None:
    # 画像をPIL Imageオブジェクトに変換
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # 予測を実行
    with st.spinner("モデルが予測中です..."):
        # モデルの予測
        results = model.predict(image)

        # 検出結果をプロットした画像を表示
        # YOLOv8のplot()メソッドを使用すると、検出された物体に自動的に枠とラベルが描画されます。
        plotted_image = results[0].plot()
        st.image(plotted_image, caption="検出結果", use_column_width=True)
        
        # 予測結果を処理
        detected_symbols = []
        if results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)
                detected_symbols.append((class_name, confidence))
        
        # 信頼度の閾値を設定
        CONFIDENCE_THRESHOLD = 0.7

        if detected_symbols:
            # 信頼度が最も高いものを結果として採用
            best_symbol, best_confidence = max(detected_symbols, key=lambda item: item[1])

            if best_confidence >= CONFIDENCE_THRESHOLD:
                st.subheader("👕 洗濯結果")
                st.write(f"**識別結果:** `{best_symbol}` (信頼度: {best_confidence:.2f})")
                
                # 洗濯注意点の表示
                if "LD_OK" in best_symbol:
                    st.success("✅ 洗濯OKです。")
                elif "HW_OK" in best_symbol:
                    st.warning("⚠️ 手洗いOKです。")
                elif "LD_NG" in best_symbol:
                    st.error("🚫 洗濯NGです。クリーニングを推奨します。")
                else:
                    st.info("識別結果が不明です。")

                # 天気予報の確認 (洗濯・手洗いOKの場合のみ)
                if "OK" in best_symbol:
                    st.subheader("☀️ 天気予報によるアドバイス")

                    with st.spinner("位置情報を取得し、天気予報を検索中です..."):
                        try:
                            # ユーザーのIPアドレスから位置情報を取得します。
                            g = geocoder.ip('me')
                            if g.latlng:
                                latitude, longitude = g.latlng
                                # geocoderから場所名を取得します
                                location_name = g.city or g.address or "不明な場所"
                                st.write(f"位置情報を取得しました: **{location_name}** (緯度 {latitude:.2f}, 経度 {longitude:.2f})")

                                # Open-Meteo APIを呼び出し
                                weather_url = "https://api.open-meteo.com/v1/forecast"
                                params = {
                                    "latitude": latitude,
                                    "longitude": longitude,
                                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code",
                                    "timezone": "auto"
                                }
                                response = requests.get(weather_url, params=params)
                                response.raise_for_status() # HTTPエラーをチェック
                                weather_data = response.json()
                                
                                # 天気予報から室内干しか外干しかを判断
                                if 'current' in weather_data:
                                    current_weather = weather_data['current']
                                    temp = current_weather.get('temperature_2m')
                                    humidity = current_weather.get('relative_humidity_2m')
                                    wind_speed = current_weather.get('wind_speed_10m')
                                    precipitation = current_weather.get('precipitation')
                                    weather_code = current_weather.get('weather_code')

                                    # 新しいロジックで乾きやすさを判断
                                    drying_info = determine_drying_conditions(temp, humidity, wind_speed, precipitation, weather_code)
                                    
                                    st.subheader(f"🧺 乾きやすさ: {drying_info['drying_status']}")
                                    st.write(drying_info['recommendation'])
                                else:
                                    st.warning("天気予報データが見つかりませんでした。")
                            else:
                                st.warning("位置情報を取得できませんでした。天気予報のアドバイスは表示されません。")
                                
                        except requests.exceptions.RequestException as e:
                            st.error(f"天気予報APIへの接続中にエラーが発生しました: {e}")
                        except Exception as e:
                            st.error(f"天気予報処理中に予期せぬエラーが発生しました: {e}")
            else:
                st.warning("画像から洗濯表示タグが検出できませんでした。")
        else:
            st.warning("画像から洗濯表示タグが検出できませんでした。")
else:
    st.info("左側のサイドバーから画像を入力してください。")

st.markdown("---")
st.markdown("※ 本アプリはデモンストレーション用です。正確な洗濯情報は必ず製品のタグをご確認ください。")
st.markdown("※ `geocoder`ライブラリは、ユーザーのIPアドレスに基づいた位置情報を取得します。プライバシーに関する懸念がある場合は、この機能をオフにするか、より正確な位置情報取得方法を検討してください。")
