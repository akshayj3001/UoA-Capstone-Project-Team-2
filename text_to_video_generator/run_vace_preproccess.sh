#------------------------ Gadio ------------------------#
python vace/gradios/vace_preproccess_demo.py

#------------------------ Video ------------------------#
python vace/vace_preproccess.py --task depth --video assets/videos/test.mp4
python vace/vace_preproccess.py --task flow --video assets/videos/test.mp4
python vace/vace_preproccess.py --task gray --video assets/videos/test.mp4
python vace/vace_preproccess.py --task pose --video assets/videos/test.mp4
python vace/vace_preproccess.py --task scribble --video assets/videos/test.mp4
python vace/vace_preproccess.py --task frameref --mode firstframe --image assets/images/test.jpg
python vace/vace_preproccess.py --task frameref --mode lastframe --expand_num 55 --image assets/images/test.jpg
python vace/vace_preproccess.py --task frameref --mode firstlastframe --image assets/images/test.jpg,assets/images/test2.jpg
python vace/vace_preproccess.py --task clipref --mode firstclip --expand_num 66 --video assets/videos/test.mp4
python vace/vace_preproccess.py --task clipref --mode lastclip --expand_num 55 --video assets/videos/test.mp4
python vace/vace_preproccess.py --task clipref --mode firstlastclip --video assets/videos/test.mp4,assets/videos/test2.mp4
python vace/vace_preproccess.py --task inpainting --mode salient --video assets/videos/test.mp4
python vace/vace_preproccess.py --task inpainting --mode mask --mask assets/masks/test.png --video assets/videos/test.mp4
python vace/vace_preproccess.py --task inpainting --mode bbox --bbox 50,50,550,700 --video assets/videos/test.mp4
python vace/vace_preproccess.py --task inpainting --mode salientmasktrack --video assets/videos/test.mp4
python vace/vace_preproccess.py --task inpainting --mode salientbboxtrack --video assets/videos/test.mp4
python vace/vace_preproccess.py --task inpainting --mode masktrack --mask assets/masks/test.png --video assets/videos/test.mp4
python vace/vace_preproccess.py --task inpainting --mode bboxtrack --bbox 50,50,550,700 --video assets/videos/test.mp4
python vace/vace_preproccess.py --task inpainting --mode label --label cat --video assets/videos/test.mp4
python vace/vace_preproccess.py --task inpainting --mode caption --caption 'boxing glove' --video assets/videos/test.mp4
python vace/vace_preproccess.py --task outpainting --video assets/videos/test.mp4
python vace/vace_preproccess.py --task outpainting --direction 'up,down,left,right' --expand_ratio 0.5 --video assets/videos/test.mp4
python vace/vace_preproccess.py --task layout_bbox --bbox '50,50,550,700 500,150,750,700' --label 'person'
python vace/vace_preproccess.py --task layout_track --mode masktrack --mask assets/masks/test.png  --label 'cat' --video assets/videos/test.mp4
python vace/vace_preproccess.py --task layout_track --mode bboxtrack --bbox '50,50,550,700' --label 'cat' --video assets/videos/test.mp4
python vace/vace_preproccess.py --task layout_track --mode label --label 'cat' --maskaug_mode hull_expand --maskaug_ratio 0.1  --video assets/videos/test.mp4
python vace/vace_preproccess.py --task layout_track --mode caption --caption 'boxing glove' --maskaug_mode bbox --video assets/videos/test.mp4 --label 'glove'

#------------------------ Image ------------------------#
python vace/vace_preproccess.py --task image_face --image assets/images/test3.jpg
python vace/vace_preproccess.py --task image_salient --image assets/images/test.jpg
python vace/vace_preproccess.py --task image_inpainting --mode 'salientbboxtrack' --image assets/images/test2.jpg
python vace/vace_preproccess.py --task image_inpainting --mode 'salientmasktrack' --maskaug_mode hull_expand --maskaug_ratio 0.3  --image assets/images/test2.jpg
python vace/vace_preproccess.py --task image_reference --mode plain --image assets/images/test.jpg
python vace/vace_preproccess.py --task image_reference --mode salient --image assets/images/test.jpg
python vace/vace_preproccess.py --task image_reference --mode mask --mask assets/masks/test2.png --image assets/images/test.jpg
python vace/vace_preproccess.py --task image_reference --mode bbox --bbox 0,264,338,636 --image assets/images/test.jpg
python vace/vace_preproccess.py --task image_reference --mode salientmasktrack --image assets/images/test.jpg   # easyway, recommend
python vace/vace_preproccess.py --task image_reference --mode salientbboxtrack --bbox 0,264,338,636 --maskaug_mode original_expand --maskaug_ratio 0.2 --image assets/images/test.jpg
python vace/vace_preproccess.py --task image_reference --mode masktrack --mask assets/masks/test2.png --image assets/images/test.jpg
python vace/vace_preproccess.py --task image_reference --mode bboxtrack --bbox 0,264,338,636 --image assets/images/test.jpg
python vace/vace_preproccess.py --task image_reference --mode label --label 'cat' --image assets/images/test.jpg
python vace/vace_preproccess.py --task image_reference --mode caption --caption 'flower' --maskaug_mode bbox --maskaug_ratio 0.3 --image assets/images/test.jpg

#------------------------ Composition ------------------------#
python vace/vace_preproccess.py --task reference_anything --mode salientmasktrack --image assets/images/test.jpg
python vace/vace_preproccess.py --task reference_anything --mode salientbboxtrack --image assets/images/test.jpg,assets/images/test2.jpg
python vace/vace_preproccess.py --task animate_anything --mode salientbboxtrack --video assets/videos/test.mp4 --image assets/images/test.jpg
python vace/vace_preproccess.py --task swap_anything --mode salientmasktrack --video assets/videos/test.mp4 --image assets/images/test.jpg
python vace/vace_preproccess.py --task swap_anything --mode label,salientbboxtrack --label 'cat' --maskaug_mode bbox --maskaug_ratio 0.3 --video assets/videos/test.mp4 --image assets/images/test.jpg
python vace/vace_preproccess.py --task swap_anything --mode label,plain --label 'cat' --maskaug_mode bbox --maskaug_ratio 0.3 --video assets/videos/test.mp4 --image assets/images/test.jpg
python vace/vace_preproccess.py --task expand_anything --mode salientbboxtrack --direction 'left,right' --expand_ratio 0.5 --expand_num 80 --image assets/images/test.jpg,assets/images/test2.jpg
python vace/vace_preproccess.py --task expand_anything --mode firstframe,plain --direction 'left,right' --expand_ratio 0.5 --expand_num 80 --image assets/images/test.jpg,assets/images/test2.jpg
python vace/vace_preproccess.py --task move_anything --bbox '0,264,338,636 400,264,538,636' --expand_num 80 --label 'cat' --image assets/images/test.jpg
