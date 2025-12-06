import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
# Load the model with custom checkpoint path
# model = build_sam3_image_model(
#     checkpoint_path="/root/autodl-tmp/sam_model_pt/sam3.pt",
#     load_from_HF=False
# )
# processor = Sam3Processor(model)
# # Load an image
# image = Image.open("/root/autodl-tmp/sam3/assets/images/truck.jpg")
# inference_state = processor.set_image(image)
# # Prompt the model with text
# output = processor.set_text_prompt(state=inference_state, prompt="car")

# # Get the masks, bounding boxes, and scores
# masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
# plot_results(image, inference_state, output_file="/root/autodl-tmp/sam3/inference_result.jpg")# 需修改



#################################### For Video ####################################
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor(
    checkpoint_path="/root/autodl-tmp/sam_model_pt/sam3.pt"
)
video_path = "/root/autodl-tmp/sam3/assets/videos/bedroom.mp4" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="children",
    )
)
output = response["outputs"]
frame_index = response["frame_index"]

# 从视频中读取对应帧用于可视化
import cv2
from sam3.visualization_utils import save_masklet_image

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # 定位到指定帧
ret, frame = cap.read()
cap.release()

if ret:
    # 将 BGR 转换为 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 保存结果图片
    output_image_path = "/root/autodl-tmp/sam3/frame_0_result.jpg"
    save_masklet_image(
        frame_rgb, 
        output, 
        out_path=output_image_path,
        frame_idx=frame_index,
        alpha=0.5  # 掩码透明度
    )
    print(f"结果已保存到: {output_image_path}")
else:
    print("无法读取视频帧")