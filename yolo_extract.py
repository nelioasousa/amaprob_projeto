from ultralytics import YOLO

model = YOLO('yolo11n.pt')

embeddings = model.embed('zidane.jpg')[0].numpy()

print(embeddings.shape)
