# from ultralytics import YOLO
#
# # 加载模型
# model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # 从YAML构建并转移权重
#
# if __name__ == '__main__':
#     # 训练模型
#     results = model.train(data='mydata.yaml', epochs=5, imgsz=512)
#
#     metrics = model.val()
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
# model = YOLO("weights/yolov8n.pt")  # 加载预训练模型（推荐用于训练）

# Use the model
# results = model.train(data="ultralytics/datasets/rain.yaml", epochs=20, batch=-1)  # 训练模型
if __name__ == '__main__':
    # Use the model
    results = model.train(data="D:/pycharm_project/pythonProject8/ultralytics/datasets/mydata.yaml", epochs=5, batch=2)  # 训练模型
    results = model.val()
    results = model("自己的验证图片")
    success = YOLO("yolov8n.pt").export(format="onnx")
#
#
# from ultralytics import YOLO
#
# # Load a model
# # model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
#
# # Use the model
# model.train(data="D:/pycharm_project/pythonProject8/ultralytics/datasets/mydata.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# path = model.export(format="onnx")  # export the model to ONNX format

