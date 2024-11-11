"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Mon Nov 11 2024
*  File : GroundedSamMgr.py
******************************************* -->

"""
from grounded_sam_demo import load_model, get_grounding_output, load_image
import os 
from duration import Duration
import glob


class App(object):
    def __init__(self):
        return
    def detect(self, image):
        tk = Duration()
        boxes_filt, pred_phrases = get_grounding_output(
            self.model, image, self.text_prompt, self.box_threshold, self.text_threshold, device=self.device)
        print(f"dino {tk.end()}")
        return

    def run(self):
        args = lambda: None
        args.config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        args.grounded_checkpoint = "./weights/groundingdino_swint_ogc.pth"
        args.sam_version = "vit_h"
        args.sam_checkpoint = "./weights/sam_vit_h_4b8939.pth"
        # args.input_image = "/media/levin/DATA/zf/nerf/2024_0601/scenes/5/rgb/es81_sur_back/rgb_00038_sur_back.jpg"
        args.input_image = "/media/levin/DATA/zf/nerf/2024_0601/scenes/5/rgb/es81_sur_back"
        args.output_dir = "outputs"
        args.box_threshold = 0.3
        args.text_threshold = 0.25
        args.text_prompt = "Tree . vegetation . road . building . sky . vehicle . pedestrian . curb . pole . traffic cone ."
        args.device = "cuda"

        args.sam_hq_checkpoint = "./weights/sam_hq_vit_h.pth"
        args.use_sam_hq = False
        args.bert_base_uncased_path = None


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


        # load model
        self.model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=self.device)

        imgs = []
        if os.path.isdir(image_path):
            imgs = glob.glob(f"{image_path}/*.jpg")
            imgs.sort()
        else:
            imgs = [image_path]

        for img_path in imgs:
            # load image
            _image_pil, image = load_image(img_path)
            self.detect(image)

        return 

if __name__ == "__main__":   
    obj= App()
    obj.run()
