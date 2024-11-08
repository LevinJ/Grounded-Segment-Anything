from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2


CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./weights/groundingdino_swint_ogc.pth"
DEVICE = "cuda"
IMAGE_PATH = "assets/rgb_00687_sur_back.jpg"
# TEXT_PROMPT = "Horse. Clouds. Grasses. Sky. Hill."
TEXT_PROMPT = "speed bumps on the road."
# BOX_TRESHOLD = 0.35
# TEXT_TRESHOLD = 0.25

BOX_TRESHOLD = 0.2
TEXT_TRESHOLD = 0.2
FP16_INFERENCE = True

image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

if FP16_INFERENCE:
    image = image.half()
    model = model.half()

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)
cv2.imshow("din0", annotated_frame)
cv2.waitKey(0) 