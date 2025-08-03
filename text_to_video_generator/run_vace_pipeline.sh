#------------------------ Pipeline ------------------------#
# extension firstframe
python vace/vace_pipeline.py --base wan --task frameref --mode firstframe --image "benchmarks/VACE-Benchmark/assets/examples/firstframe/ori_image_1.png" --prompt "纪实摄影风格，前景是一位中国越野爱好者坐在越野车上，手持车载电台正在进行通联。他五官清晰，表情专注，眼神坚定地望向前方。越野车停在户外，车身略显脏污，显示出经历过的艰难路况。镜头从车外缓缓拉近，最后定格在人物的面部特写上，展现出他的坚定与热情。中景到近景，动态镜头运镜。"

# repainting inpainting
python vace/vace_pipeline.py --base wan --task inpainting --mode salientmasktrack --maskaug_mode original_expand --maskaug_ratio 0.5 --video "benchmarks/VACE-Benchmark/assets/examples/inpainting/ori_video.mp4" --prompt "一只巨大的金色凤凰从繁华的城市上空展翅飞过，羽毛如火焰般璀璨，闪烁着温暖的光辉，翅膀雄伟地展开。凤凰高昂着头，目光炯炯，轻轻扇动翅膀，散发出淡淡的光芒。下方是熙熙攘攘的市中心，人群惊叹，车水马龙，红蓝两色的霓虹灯在夜空下闪烁。镜头俯视城市街道，捕捉这一壮丽的景象，营造出既神秘又辉煌的氛围。"

# repainting outpainting
python vace/vace_pipeline.py --base wan --task outpainting --direction 'up,down,left,right' --expand_ratio 0.3 --video "benchmarks/VACE-Benchmark/assets/examples/outpainting/ori_video.mp4" --prompt "赛博朋克风格，无人机俯瞰视角下的现代西安城墙，镜头穿过永宁门时泛起金色涟漪，城墙砖块化作数据流重组为唐代长安城。周围的街道上流动的人群和飞驰的机械交通工具交织在一起，现代与古代的交融，城墙上的灯光闪烁，形成时空隧道的效果。全息投影技术展现历史变迁，粒子重组特效细腻逼真。大远景逐渐过渡到特写，聚焦于城门特效。"

# control depth
python vace/vace_pipeline.py --base wan --task depth --video "benchmarks/VACE-Benchmark/assets/examples/depth/ori_video.mp4" --prompt "一群年轻人在天空之城拍摄集体照。画面中，一对年轻情侣手牵手，轻声细语，相视而笑，周围是飞翔的彩色热气球和闪烁的星星，营造出浪漫的氛围。天空中，暖阳透过飘浮的云朵，洒下斑驳的光影。镜头以近景特写开始，随着情侣间的亲密互动，缓缓拉远。"

# control flow
python vace/vace_pipeline.py --base wan --task flow --video "benchmarks/VACE-Benchmark/assets/examples/flow/ori_video.mp4" --prompt "纪实摄影风格，一颗鲜红的小番茄缓缓落入盛着牛奶的玻璃杯中，溅起晶莹的水花。画面以慢镜头捕捉这一瞬间，水花在空中绽放，形成美丽的弧线。玻璃杯中的牛奶纯白，番茄的鲜红与之形成鲜明对比。背景简洁，突出主体。近景特写，垂直俯视视角，展现细节之美。"

# control gray
python vace/vace_pipeline.py --base wan --task gray --video "benchmarks/VACE-Benchmark/assets/examples/gray/ori_video.mp4" --prompt "镜头缓缓向右平移，身穿淡黄色坎肩长裙的长发女孩面对镜头露出灿烂的漏齿微笑。她的长发随风轻扬，眼神明亮而充满活力。背景是秋天红色和黄色的树叶，阳光透过树叶的缝隙洒下斑驳光影，营造出温馨自然的氛围。画面风格清新自然，仿佛夏日午后的一抹清凉。中景人像，强调自然光效和细腻的皮肤质感。"

# control pose
python vace/vace_pipeline.py --base wan --task pose --video "benchmarks/VACE-Benchmark/assets/examples/pose/ori_video.mp4" --prompt "在一个热带的庆祝派对上，一家人围坐在椰子树下的长桌旁。桌上摆满了异国风味的美食。长辈们愉悦地交谈，年轻人兴奋地举杯碰撞，孩子们在沙滩上欢乐奔跑。背景中是湛蓝的海洋和明亮的阳光，营造出轻松的气氛。镜头以动态中景捕捉每个开心的瞬间，温暖的阳光映照着他们幸福的面庞。"

# control scribble
python vace/vace_pipeline.py --base wan --task scribble --video "benchmarks/VACE-Benchmark/assets/examples/scribble/ori_video.mp4" --prompt "画面中荧光色彩的无人机从极低空高速掠过超现实主义风格的西安古城墙，尘埃反射着阳光。镜头快速切换至城墙上的砖石特写，阳光温暖地洒落，勾勒出每一块砖块的细腻纹理。整体画质清晰华丽，运镜流畅如水。"

# control layout
python vace/vace_pipeline.py --base wan --task layout_track --mode bboxtrack --bbox '54,200,614,448' --maskaug_mode bbox_expand --maskaug_ratio 0.2 --label 'bird' --video "benchmarks/VACE-Benchmark/assets/examples/layout/ori_video.mp4" --prompt "视频展示了一只成鸟在树枝上的巢中喂养它的幼鸟。成鸟在喂食的过程中，幼鸟张开嘴巴等待食物。随后，成鸟飞走，幼鸟继续等待。成鸟再次飞回，带回食物喂养幼鸟。整个视频的拍摄角度固定，聚焦于巢穴和鸟类的互动，背景是模糊的绿色植被，强调了鸟类的自然行为和生态环境。"
