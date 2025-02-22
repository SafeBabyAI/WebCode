"""
가장 첫번째 버전

######## 모델 파일 경로 확인 ########
ex) WebCode-main/model/best_model.pth

pip install gradio 
pip install opencv-python   
pip install numpy   
pip install torch   
pip install torchvision     
"""

import gradio as gr
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(weights=None) # 사전 학습된 가중치를 사용하지 않고 resnet50 구조만 로드

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 1000개 → 2개 클래스 분류로 변경

######## 모델 파일 경로 확인 ########
model.load_state_dict(torch.load("model/best_model.pth", map_location=device, weights_only=True))  # 학습된 ResNet50 모델 로드, weights_only=True → 모델의 구조가 아니라 가중치만 로드
model.to(device)
model.eval() # 평가 모드로 설정

def detect_risk(frame):
    """이미지에서 아기 위험 여부 감지하는 함수"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))  # 모델 입력 크기에 맞게 조정
    img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device) # 배치 차원 추가 ((C, H, W) → (1, C, H, W))
    
    with torch.no_grad():
        output = model(img_tensor)  # 모델 출력
        probabilities = torch.softmax(output, dim=1)  # softmax로 확률 변환
        print(probabilities)
        risk_prob = probabilities[0, 0].item() # 0번 클래스(안전)의 확률 값만 추출

    alert_text = "Dangerous!" if risk_prob > 0.5 else "Safe:)"
    color = (0, 0, 255) if risk_prob > 0.5 else (0, 255, 0)
    # cv2.putText(img, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    return img, alert_text

def process_video(frame):
    #frame = cv2.resize(frame, (640, 480))
    img, alert = detect_risk(frame)
    return img, alert

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Image(sources=["webcam"], streaming=True, elem_id="input-frame"),
    outputs=[
        gr.Image(label="AI 감지 결과", elem_id="output-frame"),
        gr.Textbox(label="알람 상태")
    ],
    live=True,
    css="""
    /* 기본 제공되는 버튼 숨기기 */
    /*  #component-8 { display: none !important; } 'Clear' 버튼 숨김 */
    #component-10 { display: none !important; } /* 'Clear' 버튼 숨김 */
    
    /* 웹캠과 감지 결과 창 높이 동일하게 설정 */
    #input-frame, #output-frame {
        height: 480px !important;
    }
    """
)

demo.launch()
