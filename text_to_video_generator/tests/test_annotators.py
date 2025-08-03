# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest
import numpy as np
from PIL import Image

from vace.annotators.utils import read_video_frames
from vace.annotators.utils import save_one_video

class AnnotatorTest(unittest.TestCase):
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.save_dir = './cache/test_annotator'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # load test image
        self.image_path = './assets/images/test.jpg'
        self.image = Image.open(self.image_path).convert('RGB')
        # load test video
        self.video_path = './assets/videos/test.mp4'
        self.frames = read_video_frames(self.video_path)

    def tearDown(self):
        super().tearDown()

    @unittest.skip('')
    def test_annotator_gray_image(self):
        from vace.annotators.gray import GrayAnnotator
        cfg_dict = {}
        anno_ins = GrayAnnotator(cfg_dict)
        anno_image = anno_ins.forward(np.array(self.image))
        save_path = os.path.join(self.save_dir, 'test_gray_image.png')
        Image.fromarray(anno_image).save(save_path)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_gray_video(self):
        from vace.annotators.gray import GrayAnnotator
        cfg_dict = {}
        anno_ins = GrayAnnotator(cfg_dict)
        ret_frames = []
        for frame in self.frames:
            anno_frame = anno_ins.forward(np.array(frame))
            ret_frames.append(anno_frame)
        save_path = os.path.join(self.save_dir, 'test_gray_video.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_gray_video_2(self):
        from vace.annotators.gray import GrayVideoAnnotator
        cfg_dict = {}
        anno_ins = GrayVideoAnnotator(cfg_dict)
        ret_frames = anno_ins.forward(self.frames)
        save_path = os.path.join(self.save_dir, 'test_gray_video_2.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))


    @unittest.skip('')
    def test_annotator_pose_image(self):
        from vace.annotators.pose import PoseBodyFaceAnnotator
        cfg_dict = {
            "DETECTION_MODEL": "models/VACE-Annotators/pose/yolox_l.onnx",
            "POSE_MODEL": "models/VACE-Annotators/pose/dw-ll_ucoco_384.onnx",
            "RESIZE_SIZE": 1024
        }
        anno_ins = PoseBodyFaceAnnotator(cfg_dict)
        anno_image = anno_ins.forward(np.array(self.image))
        save_path = os.path.join(self.save_dir, 'test_pose_image.png')
        Image.fromarray(anno_image).save(save_path)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_pose_video(self):
        from vace.annotators.pose import PoseBodyFaceAnnotator
        cfg_dict = {
            "DETECTION_MODEL": "models/VACE-Annotators/pose/yolox_l.onnx",
            "POSE_MODEL": "models/VACE-Annotators/pose/dw-ll_ucoco_384.onnx",
            "RESIZE_SIZE": 1024
        }
        anno_ins = PoseBodyFaceAnnotator(cfg_dict)
        ret_frames = []
        for frame in self.frames:
            anno_frame = anno_ins.forward(np.array(frame))
            ret_frames.append(anno_frame)
        save_path = os.path.join(self.save_dir, 'test_pose_video.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_pose_video_2(self):
        from vace.annotators.pose import PoseBodyFaceVideoAnnotator
        cfg_dict = {
            "DETECTION_MODEL": "models/VACE-Annotators/pose/yolox_l.onnx",
            "POSE_MODEL": "models/VACE-Annotators/pose/dw-ll_ucoco_384.onnx",
            "RESIZE_SIZE": 1024
        }
        anno_ins = PoseBodyFaceVideoAnnotator(cfg_dict)
        ret_frames = anno_ins.forward(self.frames)
        save_path = os.path.join(self.save_dir, 'test_pose_video_2.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    # @unittest.skip('')
    def test_annotator_depth_image(self):
        from vace.annotators.depth import DepthAnnotator
        cfg_dict = {
            "PRETRAINED_MODEL": "models/VACE-Annotators/depth/depth_anything_v2_vitl.pth"
        }
        anno_ins = DepthAnnotator(cfg_dict)
        anno_image = anno_ins.forward(np.array(self.image))
        save_path = os.path.join(self.save_dir, 'test_depth_image.png')
        Image.fromarray(anno_image).save(save_path)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    # @unittest.skip('')
    def test_annotator_depth_video(self):
        from vace.annotators.depth import DepthAnnotator
        cfg_dict = {
            "PRETRAINED_MODEL": "models/VACE-Annotators/depth/depth_anything_v2_vitl.pth"
        }
        anno_ins = DepthAnnotator(cfg_dict)
        ret_frames = []
        for frame in self.frames:
            anno_frame = anno_ins.forward(np.array(frame))
            ret_frames.append(anno_frame)
        save_path = os.path.join(self.save_dir, 'test_depth_video.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_depth_video_2(self):
        from vace.annotators.depth import DepthVideoAnnotator
        cfg_dict = {
            "PRETRAINED_MODEL": "models/VACE-Annotators/depth/dpt_hybrid-midas-501f0c75.pt"
        }
        anno_ins = DepthVideoAnnotator(cfg_dict)
        ret_frames = anno_ins.forward(self.frames)
        save_path = os.path.join(self.save_dir, 'test_depth_video_2.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_scribble_image(self):
        from vace.annotators.scribble import ScribbleAnnotator
        cfg_dict = {
            "PRETRAINED_MODEL": "models/VACE-Annotators/scribble/anime_style/netG_A_latest.pth"
        }
        anno_ins = ScribbleAnnotator(cfg_dict)
        anno_image = anno_ins.forward(np.array(self.image))
        save_path = os.path.join(self.save_dir, 'test_scribble_image.png')
        Image.fromarray(anno_image).save(save_path)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_scribble_video(self):
        from vace.annotators.scribble import ScribbleAnnotator
        cfg_dict = {
            "PRETRAINED_MODEL": "models/VACE-Annotators/scribble/anime_style/netG_A_latest.pth"
        }
        anno_ins = ScribbleAnnotator(cfg_dict)
        ret_frames = []
        for frame in self.frames:
            anno_frame = anno_ins.forward(np.array(frame))
            ret_frames.append(anno_frame)
        save_path = os.path.join(self.save_dir, 'test_scribble_video.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_scribble_video_2(self):
        from vace.annotators.scribble import ScribbleVideoAnnotator
        cfg_dict = {
            "PRETRAINED_MODEL": "models/VACE-Annotators/scribble/anime_style/netG_A_latest.pth"
        }
        anno_ins = ScribbleVideoAnnotator(cfg_dict)
        ret_frames = anno_ins.forward(self.frames)
        save_path = os.path.join(self.save_dir, 'test_scribble_video_2.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_flow_video(self):
        from vace.annotators.flow import FlowVisAnnotator
        cfg_dict = {
            "PRETRAINED_MODEL": "models/VACE-Annotators/flow/raft-things.pth"
        }
        anno_ins = FlowVisAnnotator(cfg_dict)
        ret_frames = anno_ins.forward(self.frames)
        save_path = os.path.join(self.save_dir, 'test_flow_video.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_frameref_video_1(self):
        from vace.annotators.frameref import FrameRefExtractAnnotator
        cfg_dict = {
            "REF_CFG": [{"mode": "first", "proba": 0.1},
                       {"mode": "last", "proba": 0.1},
                       {"mode": "firstlast", "proba": 0.1},
                       {"mode": "random", "proba": 0.1}],
        }
        anno_ins = FrameRefExtractAnnotator(cfg_dict)
        ret_frames, ret_masks = anno_ins.forward(self.frames, ref_num=10)
        save_path = os.path.join(self.save_dir, 'test_frameref_video_1.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))
        save_path = os.path.join(self.save_dir, 'test_frameref_mask_1.mp4')
        save_one_video(save_path, ret_masks, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_frameref_video_2(self):
        from vace.annotators.frameref import FrameRefExpandAnnotator
        cfg_dict = {}
        anno_ins = FrameRefExpandAnnotator(cfg_dict)
        ret_frames, ret_masks = anno_ins.forward(frames=self.frames, mode='lastclip', expand_num=50)
        save_path = os.path.join(self.save_dir, 'test_frameref_video_2.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))
        save_path = os.path.join(self.save_dir, 'test_frameref_mask_2.mp4')
        save_one_video(save_path, ret_masks, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))


    @unittest.skip('')
    def test_annotator_outpainting_1(self):
        from vace.annotators.outpainting import OutpaintingAnnotator
        cfg_dict = {
            "RETURN_MASK": True,
            "KEEP_PADDING_RATIO": 1,
            "MASK_COLOR": "gray"
        }
        anno_ins = OutpaintingAnnotator(cfg_dict)
        ret_data = anno_ins.forward(self.image, direction=['right', 'up', 'down'], expand_ratio=0.5)
        save_path = os.path.join(self.save_dir, 'test_outpainting_image.png')
        Image.fromarray(ret_data['image']).save(save_path)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))
        save_path = os.path.join(self.save_dir, 'test_outpainting_mask.png')
        Image.fromarray(ret_data['mask']).save(save_path)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_outpainting_video_1(self):
        from vace.annotators.outpainting import OutpaintingVideoAnnotator
        cfg_dict = {
            "RETURN_MASK": True,
            "KEEP_PADDING_RATIO": 1,
            "MASK_COLOR": "gray"
        }
        anno_ins = OutpaintingVideoAnnotator(cfg_dict)
        ret_data = anno_ins.forward(frames=self.frames, direction=['right', 'up', 'down'], expand_ratio=0.5)
        save_path = os.path.join(self.save_dir, 'test_outpainting_video_1.mp4')
        save_one_video(save_path, ret_data['frames'], fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))
        save_path = os.path.join(self.save_dir, 'test_outpainting_mask_1.mp4')
        save_one_video(save_path, ret_data['masks'], fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_outpainting_inner_1(self):
        from vace.annotators.outpainting import OutpaintingInnerAnnotator
        cfg_dict = {
            "RETURN_MASK": True,
            "KEEP_PADDING_RATIO": 1,
            "MASK_COLOR": "gray"
        }
        anno_ins = OutpaintingInnerAnnotator(cfg_dict)
        ret_data = anno_ins.forward(self.image, direction=['right', 'up', 'down'], expand_ratio=0.15)
        save_path = os.path.join(self.save_dir, 'test_outpainting_inner_image.png')
        Image.fromarray(ret_data['image']).save(save_path)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))
        save_path = os.path.join(self.save_dir, 'test_outpainting_inner_mask.png')
        Image.fromarray(ret_data['mask']).save(save_path)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_outpainting_inner_video_1(self):
        from vace.annotators.outpainting import OutpaintingInnerVideoAnnotator
        cfg_dict = {
            "RETURN_MASK": True,
            "KEEP_PADDING_RATIO": 1,
            "MASK_COLOR": "gray"
        }
        anno_ins = OutpaintingInnerVideoAnnotator(cfg_dict)
        ret_data = anno_ins.forward(self.frames, direction=['right', 'up', 'down'], expand_ratio=0.15)
        save_path = os.path.join(self.save_dir, 'test_outpainting_inner_video_1.mp4')
        save_one_video(save_path, ret_data['frames'], fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))
        save_path = os.path.join(self.save_dir, 'test_outpainting_inner_mask_1.mp4')
        save_one_video(save_path, ret_data['masks'], fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_salient(self):
        from vace.annotators.salient import SalientAnnotator
        cfg_dict = {
            "PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt",
        }
        anno_ins = SalientAnnotator(cfg_dict)
        ret_data = anno_ins.forward(self.image)
        save_path = os.path.join(self.save_dir, 'test_salient_image.png')
        Image.fromarray(ret_data).save(save_path)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_salient_video(self):
        from vace.annotators.salient import SalientVideoAnnotator
        cfg_dict = {
            "PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt",
        }
        anno_ins = SalientVideoAnnotator(cfg_dict)
        ret_frames = anno_ins.forward(self.frames)
        save_path = os.path.join(self.save_dir, 'test_salient_video.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_layout_video(self):
        from vace.annotators.layout import LayoutBboxAnnotator
        cfg_dict = {
            "RAM_TAG_COLOR_PATH": "models/VACE-Annotators/layout/ram_tag_color_list.txt",
        }
        anno_ins = LayoutBboxAnnotator(cfg_dict)
        ret_frames = anno_ins.forward(bbox=[(544, 288, 744, 680), (1112, 240, 1280, 712)], frame_size=(720, 1280), num_frames=49, label='person')
        save_path = os.path.join(self.save_dir, 'test_layout_video.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_layout_mask_video(self):
        # salient
        from vace.annotators.salient import SalientVideoAnnotator
        cfg_dict = {
            "PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt",
        }
        anno_ins = SalientVideoAnnotator(cfg_dict)
        salient_frames = anno_ins.forward(self.frames)

        # mask layout
        from vace.annotators.layout import LayoutMaskAnnotator
        cfg_dict = {
            "RAM_TAG_COLOR_PATH": "models/VACE-Annotators/layout/ram_tag_color_list.txt",
        }
        anno_ins = LayoutMaskAnnotator(cfg_dict)
        ret_frames = anno_ins.forward(salient_frames, label='cat')
        save_path = os.path.join(self.save_dir, 'test_mask_layout_video.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))

    @unittest.skip('')
    def test_annotator_layout_mask_video_2(self):
        # salient
        from vace.annotators.salient import SalientVideoAnnotator
        cfg_dict = {
            "PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt",
        }
        anno_ins = SalientVideoAnnotator(cfg_dict)
        salient_frames = anno_ins.forward(self.frames)

        # mask layout
        from vace.annotators.layout import LayoutMaskAnnotator
        cfg_dict = {
            "RAM_TAG_COLOR_PATH": "models/VACE-Annotators/layout/ram_tag_color_list.txt",
            "USE_AUG": True
        }
        anno_ins = LayoutMaskAnnotator(cfg_dict)
        ret_frames = anno_ins.forward(salient_frames, label='cat', mask_cfg={'mode': 'bbox_expand'})
        save_path = os.path.join(self.save_dir, 'test_mask_layout_video_2.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))


    @unittest.skip('')
    def test_annotator_maskaug_video(self):
        # salient
        from vace.annotators.salient import SalientVideoAnnotator
        cfg_dict = {
            "PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt",
        }
        anno_ins = SalientVideoAnnotator(cfg_dict)
        salient_frames = anno_ins.forward(self.frames)

        # mask aug
        from vace.annotators.maskaug import MaskAugAnnotator
        cfg_dict = {}
        anno_ins = MaskAugAnnotator(cfg_dict)
        ret_frames = anno_ins.forward(salient_frames, mask_cfg={'mode': 'hull_expand'})
        save_path = os.path.join(self.save_dir, 'test_maskaug_video.mp4')
        save_one_video(save_path, ret_frames, fps=16)
        print(('Testing %s: %s' % (type(self).__name__, save_path)))


    @unittest.skip('')
    def test_annotator_ram(self):
        from vace.annotators.ram import RAMAnnotator
        cfg_dict = {
            "TOKENIZER_PATH": "models/VACE-Annotators/ram/bert-base-uncased",
            "PRETRAINED_MODEL": "models/VACE-Annotators/ram/ram_plus_swin_large_14m.pth",
        }
        anno_ins = RAMAnnotator(cfg_dict)
        ret_data = anno_ins.forward(self.image)
        print(ret_data)

    @unittest.skip('')
    def test_annotator_gdino_v1(self):
        from vace.annotators.gdino import GDINOAnnotator
        cfg_dict = {
            "TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
            "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
            "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth",
        }
        anno_ins = GDINOAnnotator(cfg_dict)
        ret_data = anno_ins.forward(self.image, caption="a cat and a vase")
        print(ret_data)

    @unittest.skip('')
    def test_annotator_gdino_v2(self):
        from vace.annotators.gdino import GDINOAnnotator
        cfg_dict = {
            "TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
            "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
            "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth",
        }
        anno_ins = GDINOAnnotator(cfg_dict)
        ret_data = anno_ins.forward(self.image, classes=["cat", "vase"])
        print(ret_data)

    @unittest.skip('')
    def test_annotator_gdino_with_ram(self):
        from vace.annotators.gdino import GDINORAMAnnotator
        cfg_dict = {
            "RAM": {
                "TOKENIZER_PATH": "models/VACE-Annotators/ram/bert-base-uncased",
                "PRETRAINED_MODEL": "models/VACE-Annotators/ram/ram_plus_swin_large_14m.pth",
            },
            "GDINO": {
                "TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth",
            }

        }
        anno_ins = GDINORAMAnnotator(cfg_dict)
        ret_data = anno_ins.forward(self.image)
        print(ret_data)

    @unittest.skip('')
    def test_annotator_sam2(self):
        from vace.annotators.sam2 import SAM2VideoAnnotator
        from vace.annotators.utils import save_sam2_video
        cfg_dict = {
            "CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
            "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'
        }
        anno_ins = SAM2VideoAnnotator(cfg_dict)
        ret_data = anno_ins.forward(video=self.video_path, input_box=[0, 0, 640, 480])
        video_segments = ret_data['annotations']
        save_path = os.path.join(self.save_dir, 'test_sam2_video')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_sam2_video(video_path=self.video_path, video_segments=video_segments, output_video_path=save_path)
        print(save_path)


    @unittest.skip('')
    def test_annotator_sam2salient(self):
        from vace.annotators.sam2 import SAM2SalientVideoAnnotator
        from vace.annotators.utils import save_sam2_video
        cfg_dict = {
            "SALIENT": {
                "PRETRAINED_MODEL": "models/VACE-Annotators/salient/u2net.pt",
            },
            "SAM2": {
                "CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'
            }

        }
        anno_ins = SAM2SalientVideoAnnotator(cfg_dict)
        ret_data = anno_ins.forward(video=self.video_path)
        video_segments = ret_data['annotations']
        save_path = os.path.join(self.save_dir, 'test_sam2salient_video')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_sam2_video(video_path=self.video_path, video_segments=video_segments, output_video_path=save_path)
        print(save_path)


    @unittest.skip('')
    def test_annotator_sam2gdinoram_video(self):
        from vace.annotators.sam2 import SAM2GDINOVideoAnnotator
        from vace.annotators.utils import save_sam2_video
        cfg_dict = {
            "GDINO": {
                "TOKENIZER_PATH": "models/VACE-Annotators/gdino/bert-base-uncased",
                "CONFIG_PATH": "models/VACE-Annotators/gdino/GroundingDINO_SwinT_OGC_mod.py",
                "PRETRAINED_MODEL": "models/VACE-Annotators/gdino/groundingdino_swint_ogc.pth",
            },
            "SAM2": {
                "CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'
            }
        }
        anno_ins = SAM2GDINOVideoAnnotator(cfg_dict)
        ret_data = anno_ins.forward(video=self.video_path, classes='cat')
        video_segments = ret_data['annotations']
        save_path = os.path.join(self.save_dir, 'test_sam2gdino_video')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_sam2_video(video_path=self.video_path, video_segments=video_segments, output_video_path=save_path)
        print(save_path)

    @unittest.skip('')
    def test_annotator_sam2_image(self):
        from vace.annotators.sam2 import SAM2ImageAnnotator
        cfg_dict = {
            "CONFIG_PATH": 'models/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
            "PRETRAINED_MODEL": 'models/VACE-Annotators/sam2/sam2.1_hiera_large.pt'
        }
        anno_ins = SAM2ImageAnnotator(cfg_dict)
        ret_data = anno_ins.forward(image=self.image, input_box=[0, 0, 640, 480])
        print(ret_data)

    @unittest.skip('')
    def test_annotator_prompt_extend(self):
        from vace.annotators.prompt_extend import PromptExtendAnnotator
        from vace.configs.prompt_preprocess import WAN_LM_ZH_SYS_PROMPT, WAN_LM_EN_SYS_PROMPT, LTX_LM_EN_SYS_PROMPT
        cfg_dict = {
            "MODEL_NAME": "models/VACE-Annotators/llm/Qwen2.5-3B-Instruct" # "Qwen2.5_3B"
        }
        anno_ins = PromptExtendAnnotator(cfg_dict)
        ret_data = anno_ins.forward('一位男孩', system_prompt=WAN_LM_ZH_SYS_PROMPT)
        print('wan_zh:', ret_data)
        ret_data = anno_ins.forward('a boy', system_prompt=WAN_LM_EN_SYS_PROMPT)
        print('wan_en:', ret_data)
        ret_data = anno_ins.forward('a boy', system_prompt=WAN_LM_ZH_SYS_PROMPT)
        print('wan_zh en:', ret_data)
        ret_data = anno_ins.forward('a boy', system_prompt=LTX_LM_EN_SYS_PROMPT)
        print('ltx_en:', ret_data)

        from vace.annotators.utils import get_annotator
        anno_ins = get_annotator(config_type='prompt', config_task='ltx_en', return_dict=False)
        ret_data = anno_ins.forward('a boy', seed=2025)
        print('ltx_en:', ret_data)
        ret_data = anno_ins.forward('a boy')
        print('ltx_en:', ret_data)
        ret_data = anno_ins.forward('a boy', seed=2025)
        print('ltx_en:', ret_data)

    @unittest.skip('')
    def test_annotator_prompt_extend_ds(self):
        from vace.annotators.utils import get_annotator
        # export DASH_API_KEY=''
        anno_ins = get_annotator(config_type='prompt', config_task='wan_zh_ds', return_dict=False)
        ret_data = anno_ins.forward('一位男孩', seed=2025)
        print('wan_zh_ds:', ret_data)
        ret_data = anno_ins.forward('a boy', seed=2025)
        print('wan_zh_ds:', ret_data)


# ln -s your/path/annotator_models annotator_models
# PYTHONPATH=. python tests/test_annotators.py
if __name__ == '__main__':
    unittest.main()
