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

    geo_out=nerf/geo-tex

    python launch.py --config configs/geo-tex.yaml \
            --train --gpu 0 system.prompt_processor.prompt="$prompt"  use_timestamp=False \
            name=$geo_out \
            data.width=[64,192] data.height=[64,192] data.batch_size=[4,4] ${@:4}

done
