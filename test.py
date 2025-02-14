import gradio as gr
import matplotlib as plt
import requests, json
def request_iris_prediction(data_list):
    endpoint = "http://c8036432-a374-4b6f-b89d-067ef0d15b8a.koreacentral.azurecontainer.io/score"
    # method : post
    # headers
    headers = { 
        "Content-Type" : "application/json",
        "Authorization" : "Bearer VIrzntKeu94DrPK0KVx9ktjtYklFROHS"
    }
    # body
    body = {
        "Inputs": {
            "input1": data_list
        }
    }
    response = requests.post(endpoint, headers=headers, json=body)
    print(response)
    if response.status_code == 200:
        response_json = response.json()
        return response_json["Results"]["WebServiceOutput0"]
    else:
        return ""

def save_plot(data_points):
    # 센터로이드의 평균 위치를 계산하기 위한 변수 초기화
    centroid_positions = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    centroid_colors = {0: 'b', 1: 'r', 2: 'g'}  # 클러스터 색상
    
    # 데이터 포인트를 기반으로 센터로이드 위치 계산
    for point in data_points:
        assignment = point["Assignments"]
    
        # 각 클러스터별로 거리 데이터 가져오기
        for i in range(3):
            dist_key = f"DistancesToClusterCenter no.{i}"
            if dist_key in point:
                # 위치의 평균 계산
                centroid_positions[i][0] += (point["sepal_length_cm"] + point[dist_key]) / 2
                centroid_positions[i][1] += (point["sepal_width_cm"] + point[dist_key]) / 2
    
    # 평균값으로 센터로이드 위치 계산
    for i in range(3):
        centroid_positions[i][0] /= len(data_points)
        centroid_positions[i][1] /= len(data_points)
    
    plt.figure(figsize=(8, 6))
    
    point_index = 0
    # 데이터 포인트 그리기
    for point in data_points:
        point_index += 1
        plt.scatter(point["sepal_length_cm"], point["sepal_width_cm"],
                    c='b' if point["Assignments"] == 0 else 'r' if point["Assignments"] == 1 else 'g')
        plt.text(point["sepal_length_cm"], point["sepal_width_cm"], f"{point_index}")

    # 클러스터 센터로이드 그리기
    for cluster, (x, y) in centroid_positions.items():
        plt.scatter(x, y, c=centroid_colors[cluster], marker='X', s=200)

    plt.title('Data Points and Cluster Centroids')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.grid()
    plt.savefig('iris_clusters.png')
    plt.close()  # plt.show() 대신 plt.close()를 사용
    return 'iris_clusters.png'  # 현재 figure 반환


with gr.Blocks() as demo:
    gr.Markdown("# 붓꽃 예측!!!!")
    
    view_count = gr.State(1)
    data_dict = dict()
    
    def click_send():
        data_list = list(data_dict.values())
        print(data_list)
        response_data = request_iris_prediction(data_list) 
        image_path = save_plot(response_data)
        return json.dumps(response_data, indent=3), image_path
    
    def click_add(count):
        count += 1
        print("+ : {}".format(count))
        return count

    def click_delete(count):
        
        if count > 1:
            count -= 1
        print("- : {}".format(count))
        return count

    def change_data(row_index, sl, sw, pl, pw):
        data_dict.update({
            row_index: {
                "sepal_length_cm": sl,
                "sepal_width_cm": sw,
                "petal_length_cm": pl,
                "petal_width_cm": pw,
                "class": ""
            }
        })
        print(data_dict)
        pass
    
    with gr.Row():
        add_button = gr.Button("+")
        delete_button = gr.Button("-")
    
    with gr.Column():
        
        @gr.render(inputs=[view_count])
        def render_input_components(count):
            
            for i in range(0, count):
                with gr.Column():
                    gr.Markdown(value=f"Index : {i}")
                    row_index = gr.State(i)
                    with gr.Row():
                        
                        sepal_length_textbox = gr.Textbox(label="꽃받침 길이", key=f"sl-{i}")
                        sepal_width_textbox = gr.Textbox(label="꽃받침 넓이", key=f"sw-{i}")
                        petal_length_textbox = gr.Textbox(label="꽃잎 길이", key=f"pl-{i}")
                        petal_width_textbox = gr.Textbox(label="꽃잎 넓이", key=f"pw-{i}")
                        
                        sepal_length_textbox.change(fn=change_data, inputs=[row_index, sepal_length_textbox, sepal_width_textbox, petal_length_textbox, petal_width_textbox], outputs=[])
                        sepal_width_textbox.change(fn=change_data, inputs=[row_index, sepal_length_textbox, sepal_width_textbox, petal_length_textbox, petal_width_textbox], outputs=[])
                        petal_length_textbox.change(fn=change_data, inputs=[row_index, sepal_length_textbox, sepal_width_textbox, petal_length_textbox, petal_width_textbox], outputs=[])
                        petal_width_textbox.change(fn=change_data, inputs=[row_index, sepal_length_textbox, sepal_width_textbox, petal_length_textbox, petal_width_textbox], outputs=[])
    
    send_button = gr.Button("전송")
    output_textbox = gr.Textbox(label="출력")    
    output_image = gr.Image(label="Plot", interactive=False)
    
    add_button.click(fn=click_add, inputs=[view_count], outputs=[view_count])
    delete_button.click(fn=click_delete, inputs=[view_count], outputs=[view_count])
    
    send_button.click(fn=click_send, inputs=[], 
                      outputs=[output_textbox, output_image])
    
demo.launch(share=True)

