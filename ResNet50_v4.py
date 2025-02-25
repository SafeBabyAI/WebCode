"""
최종 resnet+yolo
"""

import os

import gradio as gr

from models.resnet_model import ResNetModel
from models.yolo_model import YOLODetector
# import kakaotalk

END_COUNT = 10
RISK_MESSAGE_SEND = "🚨아기를 확인해주세요🚨"
RISK_MESSAGE_TEXTBOX = "위험 상황이 지속되어 스트리밍을 중지합니다!"
risk_count = 0  # 리스크 카운트 변수

# 모델 폴더 경로
model_dir = "project\WebCode-main\model"
yolo_dir = "project\WebCode-main\model\yolo_best.pt"

def get_model_files():
    """모델 폴더 내의 파일 리스트를 가져오는 함수"""
    files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    files.sort()
    return files

file_path = os.path.join(model_dir, get_model_files()[0] if get_model_files() else "")
resnet_model = ResNetModel(file_path)
yolo_detector = YOLODetector(yolo_dir)

def update_resnet_model(selected_model):
    """선택된 모델을 로드하는 함수"""
    global resnet_model
    model_path = os.path.join(model_dir, selected_model)
    resnet_model = ResNetModel(model_path)
    return f"모델 변경 완료: {selected_model}"

def detect_risk(frame):
    global risk_count, resnet_model, yolo_detector
    #####################################
    # 허깅페이스 적용 코드
    #####################################
    # position = resnet_model.predict(frame)
    #
    # if position == "Back":
    #     risk_count += 1
    #     alert_text = "Dangerous!"
    #     print("Back")
    # else:
    #     nose_detected, mouth_detected, frame = yolo_detector.detect_nose_mouth(frame)
    #     if (not nose_detected and not mouth_detected):
    #         alert_text = "Dangerous!"
    #         print("코가 없다, 입이 없다")
    #         risk_count += 1
    #     else: 
    #         alert_text = "Safe:)"
    #         risk_count = 0
    #####################################
    # 기본 적용 코드
    #####################################
    position = resnet_model.predict(frame)
    nose_detected, mouth_detected, frame = yolo_detector.detect_nose_mouth(frame)

    if position == "Back" or (not nose_detected and not mouth_detected):
        risk_count += 1
        alert_text = "Dangerous!"
    else:
        risk_count = 0
        alert_text = "Safe:)"
    #####################################
    
    return frame, alert_text

def process_video(frame):
    global risk_count
    
    if risk_count >= END_COUNT:
        print(f"========= 위험 감지 (risk_count: {risk_count}) =========")
        alert = RISK_MESSAGE_TEXTBOX
        webcam_state = gr.update(streaming=False)
        button_state = gr.update(interactive=True)
        img = None
    else: 
        img, alert = detect_risk(frame)
        webcam_state = None
        button_state = gr.update(interactive=False)
    
    return webcam_state, button_state, img, alert

# def send_message():
#     message_state = kakaotalk.send_message(RISK_MESSAGE_SEND)
#     print(message_state)

def alert_warning(message):
    if message == RISK_MESSAGE_TEXTBOX:
        print("메시지 전송")
        # send_message()
        gr.Warning(RISK_MESSAGE_SEND, duration=10)

def click_start_button():
    global risk_count
    risk_count = 0  # 위험 감지 후 초기화
    print(f"click_start_button() (risk_count: {risk_count})")
    webcam_state = gr.update(streaming=True)
    button_state = gr.update(interactive=False)
    return webcam_state, button_state

webcam_constraints = {
    "video": {
        "width": {"ideal": 480},
        "height": {"ideal": 480},
    },
}
css = """
    .startButton{
        height: 5.6em
    }
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown("## AI 기반 아기 위험 감지 시스템")
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=get_model_files(), show_label=False, value=os.path.basename(file_path))
        model_update_status = gr.Textbox(show_label=False, interactive=False)
    
    with gr.Row():
        alert_text = gr.Textbox(label="알람 상태")
        start_button = gr.Button("재시작", interactive=False, elem_classes="startButton")
    
    with gr.Row():
        webcam = gr.Image(sources="webcam", streaming=True, height=480, webcam_constraints=webcam_constraints)
        output_img = gr.Image(label="AI 감지 결과", height=480, webcam_constraints=webcam_constraints)
    
    model_dropdown.change(fn=update_resnet_model, inputs=[model_dropdown], outputs=[model_update_status])
    webcam.stream(fn=process_video, inputs=[webcam], outputs=[webcam, start_button, output_img, alert_text])
    start_button.click(fn=click_start_button, inputs=[], outputs=[webcam, start_button])
    alert_text.change(fn=alert_warning, inputs=[alert_text], outputs=[])

demo.launch(share=True)
