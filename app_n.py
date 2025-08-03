import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Streamlitアプリのタイトル
st.title("洗濯表示マーク判別アプリ")

st.write("洗濯表示タグの画像をアップロードしてください。")

# YOLOモデルのロード
# モデルファイルのパスを適切に指定してください
try:
    model = YOLO("best_5.pt")
except Exception as e:
    st.error(f"モデルのロードに失敗しました。'best.pt'ファイルが正しいパスにあるか確認してください。エラー: {e}")
    st.stop()

# アップロードされた画像の処理
uploaded_file = st.file_uploader("画像を選択してください...", type=["jpg", "jpeg", "png"])

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
    # 推論結果をresultsとして取得
    results = model(img_np)

    # 推論結果の表示
    detected_marks = []
    washing_instructions = [] # 新しく洗濯指示を格納するリスト

    # 推論結果からバウンディングボックスとクラス情報を取得
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            class_name = model.names[cls] # クラス名をモデルから取得
            
            # 信頼度（confidence）を0.2以上のもののみ表示する例
            if conf > 0.2:
                detected_marks.append(f"• {class_name} (信頼度: {conf:.2f})")
                
                # 洗濯指示のロジックを追加
                if class_name == "HW_OK":
                    washing_instructions.append("手洗いコースを選んでください。")
                elif class_name == "LD_OK":
                    washing_instructions.append("洗濯機での洗濯ができます。")
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
