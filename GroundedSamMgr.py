"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Mon Nov 11 2024
*  File : GroundedSamMgr.py
******************************************* -->

"""
from grounded_sam_demo import load_model, get_grounding_output, load_image, show_mask, show_box
import os 
from duration import Duration
import glob
import cv2
import torch
import matplotlib.pyplot as plt
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import supervision as sv
import torchvision
from groundingdino.util.inference import Model
import numpy as np


class App(object):
    def __init__(self):
        return
    def detect_1(self, image):
        tk = Duration()
        boxes_filt, pred_phrases = get_grounding_output(
            self.model, image, self.text_prompt, self.box_threshold, self.text_threshold, device=self.device)
        print(f"dino {tk.end()}")
        #output of detection
        self.boxes_filt = boxes_filt
        self.pred_phrases = pred_phrases
        return
    def segment_1(self, image):
        predictor = self.predictor
        boxes_filt = self.boxes_filt
        device = self.device

        tk = Duration()
        predictor.set_image(image)
        
        # size = image_pil.size
        H, W = image.shape[:2]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        print(f"sam {tk.end()}")

        #output of segment
        self.image = image
        self.masks = masks
       
        return
    
    def vis_seg(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        for mask in self.masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(self.boxes_filt, self.pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
        plt.show()
        return
    def process_1(self, img_path):
        # load image
        _image_pil, image = load_image(img_path)
        self.detect_1(image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.segment_1(image)
        self.vis_seg()
        return
    def detect_2(self, image):
        CLASSES = self.CLASSES
        tk = Duration()
        # detect objects
        detections = self.model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold 
        )
        print(f"dino {tk.end()}")
       

        

        # # NMS post process
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        # nms_idx = torchvision.ops.nms(
        #     torch.from_numpy(detections.xyxy), 
        #     torch.from_numpy(detections.confidence), 
        #     self.NMS_THRESHOLD
        # ).numpy().tolist()

        # detections.xyxy = detections.xyxy[nms_idx]
        # detections.confidence = detections.confidence[nms_idx]
        # detections.class_id = detections.class_id[nms_idx]

        # print(f"After NMS: {len(detections.xyxy)} boxes")
        #output detection result
        self.detections = detections

        # save the annotated grounding dino image
        # annotate image with detections
        # box_annotator = sv.BoxAnnotator()
        # labels = [
        #     f"{CLASSES[class_id]} {confidence:0.2f}" 
        #     for _, _, confidence, class_id, _, _ 
        #     in detections]
        # annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        # cv2.imshow("din0", annotated_frame)
        # cv2.waitKey(0)
        return
    def segment2_internal(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        device = "cuda"
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(torch.tensor(xyxy), image.shape[:2]).to(device)
        masks, _, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        return masks.detach().cpu().numpy().squeeze()

       
    
    def segment_2(self, image):
        sam_predictor = self.predictor
        detections = self.detections

        tk = Duration()

        detections.mask = self.segment2_internal(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )

        print(f"sam {tk.end()}")
        return
    def vis_2(self, image, CLASSES, detections):
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        # annotated_image = mask_annotator.annotate(scene=np.zeros_like(image), detections=detections)
        annotated_image = mask_annotator.annotate(image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # print("done")
        cv2.imshow("din0", annotated_image)
        cv2.waitKey(1) 
        return
    def process_2(self, img_path):
        # load image
        image = cv2.imread(img_path)
        self.detect_2(image)
        self.segment_2(image)
        self.vis_2(image, self.CLASSES, self.detections)
        
        return
   

    def run(self):
        args = lambda: None
        args.config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        args.grounded_checkpoint = "./weights/groundingdino_swint_ogc.pth"
        
        
        
        # args.input_image = "/media/levin/DATA/zf/nerf/2024_0601/scenes/5/rgb/es81_sur_back/rgb_00038_sur_back.jpg"
        args.input_image = "/media/levin/DATA/zf/nerf/2024_0601/scenes/5/rgb/es81_sur_back"
        args.output_dir = "outputs"
        args.box_threshold = 0.2
        args.text_threshold = 0.2
        CLASSES = [  "Tree", 'vegetation', "road", "building", "sky", "vehicle", "pedestrian", "curb", "pole", "traffic cone"]
        args.text_prompt = ". ".join(CLASSES)
        args.device = "cuda"

        
        args.use_sam_hq = False

        args.sam_version = "vit_b"
        args.sam_checkpoint = "./weights/sam_vit_b_01ec64.pth"
        self.NMS_THRESHOLD = 0.8

        # args.sam_version = "vit_h"
        # args.sam_checkpoint = "./weights/sam_vit_h_4b8939.pth"

        # args.sam_version = "vit_b"
        # args.sam_hq_checkpoint = "./weights/sam_hq_vit_b.pth"

        # args.sam_version = "vit_h"
        # args.sam_hq_checkpoint = "./weights/sam_hq_vit_h.pth"

        # args.sam_version = "vit_tiny"
        # args.sam_hq_checkpoint = "./weights/sam_hq_vit_tiny.pth"
        
        args.bert_base_uncased_path = None
        if not args.use_sam_hq:
            args.sam_hq_checkpoint = None
        else:
            args.sam_checkpoint = None



        config_file = args.config  # change the path of the model config file
        grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
        sam_version = args.sam_version
        sam_checkpoint = args.sam_checkpoint
        sam_hq_checkpoint = args.sam_hq_checkpoint
        use_sam_hq = args.use_sam_hq
        image_path = args.input_image
        self.text_prompt = args.text_prompt
        output_dir = args.output_dir
        self.box_threshold = args.box_threshold
        self.text_threshold = args.text_threshold
        self.device = args.device
        bert_base_uncased_path = args.bert_base_uncased_path
        self.CLASSES = CLASSES


        use_method_1 = False
        # load model
        if use_method_1:
            self.model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=self.device)
        else:
            # Building GroundingDINO inference model
            self.model = Model(model_config_path=config_file, model_checkpoint_path=grounded_checkpoint)

        # initialize SAM
        if use_sam_hq:
            self.predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(self.device))
        else:
            self.predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(self.device))

        imgs = []
        if os.path.isdir(image_path):
            imgs = glob.glob(f"{image_path}/*.jpg")
            imgs.sort()
        else:
            imgs = [image_path]

        for img_path in imgs:
            print(img_path)
            if use_method_1:
                self.process_1(img_path)
            else:
                self.process_2(img_path)

        return 

if __name__ == "__main__":   
    obj= App()
    obj.run()
