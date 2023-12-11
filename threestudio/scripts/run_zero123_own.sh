set -e
NAME="shoe4_1"

# Phase 1 - 64x64
# TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 HF_HUB_OFFLINE=1 python launch.py --config configs/zero123_sdf.yaml --train --gpu 0 data.image_path=./load/images/${NAME}_rgba.png use_timestamp=False name=${NAME} tag=Phase1  # data.random_camera.batch_size=[1,1,1]
TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 HF_HUB_OFFLINE=1 python launch.py --config configs/zero123.yaml --train --gpu 0 data.image_path=./load/images/${NAME}_rgba.png use_timestamp=False name=${NAME} tag=Phase1 data.random_camera.batch_size=[1,1,1] system.freq.guidance_eval=10

# Phase 1.5 - 512 refine
TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 HF_HUB_OFFLINE=1 python launch.py --config configs/zero123-geometry.yaml --train --gpu 0 data.image_path=./load/images/${NAME}_rgba.png system.geometry_convert_from=./outputs/${NAME}/Phase1/ckpts/last.ckpt use_timestamp=False name=${NAME} tag=Phase1p5   system.freq.guidance_eval=10

# # Phase 2 - dreamfusion
# python launch.py --config configs/experimental/imagecondition_zero123nerf.yaml --train --gpu 0 data.image_path=./load/images/${NAME}_rgba.png system.prompt_processor.prompt="A 3D model of a friendly dragon" system.weights="outputs/${NAME}/Phase1/ckpts/last.ckpt" name=${NAME} tag=Phase2  data.random_camera.batch_size=1

# # Phase 2 - SDF + dreamfusion
# python launch.py --config configs/experimental/imagecondition_zero123nerf_refine.yaml --train --gpu 0 data.image_path=./load/images/${NAME}_rgba.png system.prompt_processor.prompt="A 3D model of a friendly dragon" system.geometry_convert_from="outputs/${NAME}/Phase1/ckpts/last.ckpt" name=${NAME} tag=Phase2_refine data.batch_size=2
