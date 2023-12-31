export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=./pretrained_models/huggingface

readarray -t prompts < <(sed 's/\n$//' $3)

for ((i=$1; i<$2; i++));
do
    echo $i. "${prompts[$i]}"
    result=$(echo "${prompts[$i]}" | tr ' ' '_')
    result=$(echo "$result" | tr -d '"')
    echo $result

    prompt="${prompts[$i]}"

    geo_out=nerf/geo-res256
    geo_refine_out=nerf/geo-refine-res256
    tex_out=nerf/tex-res256

    python launch.py --config configs/nd-mv-nerf/geo.yaml \
            --train --gpu 0 system.prompt_processor.prompt="$prompt"  use_timestamp=False \
            name=$geo_out \
            system.guidance.share_t=false \
            data.width=[64,256] data.height=[64,256] data.batch_size=[8,4] \
            trainer.max_steps=5000  ${@:4}

    python launch.py --config configs/nd-mv-nerf/geo-refine.yaml  \
            --train --gpu 0  system.prompt_processor.prompt="$prompt"  use_timestamp=False  \
            name=$geo_refine_out \
            system.geometry_convert_from=outputs/$geo_out/$result/ckpts/last.ckpt \
            system.geometry_convert_override.isosurface_threshold=10. \
            trainer.max_steps=2000 ${@:4}

    prompt="a DSLR photo of ${prompts[$i]}"

    python launch.py --config configs/nd-mv-nerf/tex.yaml  \
            name=$tex_out \
            system.prompt_processor.prompt="$prompt"  use_timestamp=False  \
            system.geometry_convert_from=outputs/$geo_refine_out/$result/ckpts/last.ckpt \
            --train --gpu 0 system.prompt_processor.prompt="$prompt"  use_timestamp=False \
            trainer.max_steps=3000 ${@:4}

done
