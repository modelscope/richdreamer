export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=./pretrained_models/huggingface
GPU_ID=$4

readarray -t prompts < <(sed 's/\n$//' $3)
for ((i=$1; i<$2; i++));
do
    echo $i. "${prompts[$i]}"
    result=$(echo "${prompts[$i]}" | tr ' ' '_')
    result=$(echo "$result" | tr -d '"')
    echo $result

    prompt="${prompts[$i]}"

    geo_out=dmtet/geo
    geo_refine_out=dmtet/geo-refine
    tex_out=dmtet/tex
    exp_root_dir=./

    rm -rf $exp_root_dir/$geo_out/$result
    rm -rf $exp_root_dir/$geo_refine_out/$result
    rm -rf $exp_root_dir/$tex_out/a_DSLR_photo_of_$result

    # step.1
    python3 launch.py --config configs/nd-mv-dmtet/geo.yaml \
            --train --gpu $GPU_ID system.prompt_processor.prompt="$prompt"  use_timestamp=False \
            name=$geo_out \
            system.anneal_normal_stone="[500, 1000]" \
            data.elevation_range="[5, 30]" \
            data.fovy_range="[40, 45]" \
            data.camera_distance_range="[0.8, 1.0]" \
            exp_root_dir=$exp_root_dir

    # step.2
    python3 launch.py --config configs/nd-mv-dmtet/geo-refine.yaml  \
            --train --gpu $GPU_ID  system.prompt_processor.prompt="$prompt"  use_timestamp=False  \
            name=$geo_refine_out \
            system.geometry_convert_from=$exp_root_dir/$geo_out/$result/ckpts/last.ckpt \
            exp_root_dir=$exp_root_dir ${@:4}

    # step.3
    prompt="a DSLR photo of ${prompts[$i]}" # trick propose by Fantasia3D
    python3 launch.py --config configs/nd-mv-dmtet/tex.yaml  \
            name=$tex_out \
            system.prompt_processor.prompt="$prompt"  use_timestamp=False  \
            system.geometry_convert_from=$exp_root_dir/$geo_refine_out/$result/ckpts/last.ckpt \
            --train --gpu $GPU_ID system.prompt_processor.prompt="$prompt"  use_timestamp=False \
            exp_root_dir=$exp_root_dir ${@:4}

done
