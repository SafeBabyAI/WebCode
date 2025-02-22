"""
하나의 모델을 선택해 적용합니다.(드롭박스 없는 버전)

** model_dir 의 모델 파일 경로 확인 필수 **
ex) WebCode-main\model\best_model V5.pth

END_COUNT : 위험 감지를 빠르게 혹은 느리게 조정
RISK_MESSAGE_SEND : 카카오톡, 경고 팝업에 표시되는 메시지
RISK_MESSAGE_TEXTBOX : 카메라 하단 텍스트박스에 표시되는 메시지

아래는 사용한 라이브러리의 설치 명령어 리스트 입니다.   
오류나는 부분은 설치가 필요할 수 있습니다.  
커맨드창이나 파워셀에 입력하여 설치하실 수 있습니다.  
카카오톡 전송부분은 파일을 따로 업로드 하지 않아 모두 주석처리 하였습니다.

pip install gradio 
pip install opencv-python   
pip install numpy   
pip install torch   
pip install torchvision   
"""

import os
import gradio as gr
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
# import kakaotalk


END_COUNT = 5
RISK_MESSAGE_SEND = "🚨아기를 확인해주세요🚨"
RISK_MESSAGE_TEXTBOX = "위험 상황이 지속되어 스트리밍을 중지합니다!"
risk_count = 0 # 리스크 카운트 변수
device = "cuda" if torch.cuda.is_available() else "cpu"

## 사용자 환경에서 경로 재확인 필요 ##
# file_path = r"model\best_model.pth" # 2개 클래스 분류
# file_path = r"model\best_model V3.pth" # 두번째 우수
file_path = r"WebCode-main\best_model V5.pth" # 가장 우수
# file_path = r"model\best_model V6.pth" # 배경 인식률이 90% X
# file_path = r"model\best_model V7.pth" # 동영상은 아쉽지만 사진 판단 우수


def create_resnet50_model():
    model = models.resnet50(weights=None) # 사전 학습된 가중치를 사용하지 않고 resnet50 구조만 로드
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 3)  # 1000개 → 3개 클래스 분류로 변경
    model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))  # 학습된 ResNet50 모델 로드, weights_only=True → 모델의 구조가 아니라 가중치만 로드
    model.to(device)
    model.eval() # 평가 모드로 설정
    gr.Info("resnet 시작")
    return model

model = create_resnet50_model()

"""이미지에서 아기 위험 여부 감지하는 함수"""
def detect_risk(frame):
    global risk_count, model
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))  # 모델 입력 크기에 맞게 조정
    img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device) # 배치 차원 추가 ((C, H, W) → (1, C, H, W))

    with torch.no_grad():
        output = model(img_tensor)  # 모델 출력
        probabilities = torch.softmax(output, dim=1)  # softmax로 확률 변환
        print(probabilities)
        risk_prob = probabilities[0, 0].item() # 0번 클래스(안전)의 확률 값만 추출

    if risk_prob > 0.7 :
        alert_text = "Dangerous!"
        risk_count += 1
    else :
        alert_text = "Safe:)"
        risk_count = 0

    color = (0, 0, 255) if risk_prob > 0.7 else (0, 255, 0)
    # cv2.putText(img, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    return img, alert_text

def process_video(frame):
    global risk_count  # 두 전역 변수 모두 사용
    img, alert = detect_risk(frame)
    webcam_state = None
    button_state = gr.update(interactive=False)

    if risk_count == END_COUNT:
        alert = RISK_MESSAGE_TEXTBOX
        print("=========risk_count: " + str(risk_count))
        #########################################################################
        # 아래 웹캠 부분을 주석처리하면 웹캠이 멈춤없이 계속 구동합니다.
        # 웹캠이 멈추지 않으면 경고 알림이 빨리 사라집니다. 이상현상x
        webcam_state = gr.update(streaming=False)
        #########################################################################
        button_state = gr.update(interactive=True)
        print("메시지 전송")
        # send_message()

    return img, alert, webcam_state, button_state

# def send_message():
#     message_state = kakaotalk.send_message(RISK_MESSAGE_SEND)
#     print(message_state)

def alert_warning(message):
    if message == RISK_MESSAGE_TEXTBOX:
        gr.Warning(RISK_MESSAGE_SEND, duration=10)

webcam_constraints = {
    "video": {
        "width": {"ideal" : 480},
        "height": {"ideal": 480},
    },
}
with gr.Blocks() as demo:
    gr.Markdown("## AI 기반 아기 위험 감지 시스템")
    with gr.Row():
        webcam = gr.Image(sources="webcam", streaming=True, webcam_constraints=webcam_constraints)
        output_img = gr.Image(label="AI 감지 결과", webcam_constraints=webcam_constraints)
    alert_text = gr.Textbox(label="알람 상태")
    start_button = gr.Button("시작", interactive=False)

    def start_streming():
        global webcam
        webcam.streaming = True
        start_button.interactive = False
        return webcam, start_button

    webcam.stream(fn=process_video, inputs=[webcam], outputs=[output_img, alert_text, webcam, start_button])
    start_button.click(fn=start_streming, inputs=[], outputs=[webcam, start_button])
    alert_text.change(fn=alert_warning, inputs=[alert_text], outputs=[])

demo.launch(share=True)
cv2.destroyAllWindows()
