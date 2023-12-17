set -e
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=./pretrained_models/huggingface


GPUS=$1
prompt=$2
SAVE_PATH=$3
IMG_RES=$4
SAVE_MEM=$5
exp_root_dir=$SAVE_PATH/tmp
SAVE_MODEL_PATH=$SAVE_PATH/result
mkdir -p  $exp_root_dir
mkdir -p  $SAVE_MODEL_PATH

G_TYPE="nd-multiview-diffusion-guidance"
if [ $SAVE_MEM -gt 0 ]; then
    # for saving GPU memory
    G_TYPE="none"
fi

if [ $6 ]; then
    # for debuging
    echo "$6"
    ITER=$(echo "$6" | tr 'trainer.max_steps=' ' ')
    ITER=$(echo $ITER | sed 's/ //g')
else
    ITER=3000
fi

echo "GPU: ${GPUS}"
echo "ITER: $ITER"
echo "guidance_type:  $G_TYPE"
echo "input prompt: $prompt"
result=$(echo "${prompt}" | tr ' ' '_')
result=$(echo "$result" | tr -d '"')
echo output_dir: $SAVE_PATH

geo_out=nd-mv-nerf/geo-fast
geo_refine_out=nd-mv-nerf/geo-refine-fast
tex_out=nd-mv-nerf/tex-fast

rm -rf $exp_root_dir/$geo_out/$result
rm -rf $exp_root_dir/$geo_refine_out/$result
rm -rf $exp_root_dir/$tex_out/a_DSLR_photo_of_$result


python3 launch.py --config configs/nd-mv-nerf/geo.yaml \
        --train --gpu $GPUS system.prompt_processor.prompt="$prompt"  use_timestamp=False \
        name=$geo_out \
        data.width=[64,$IMG_RES] data.height=[64,$IMG_RES] data.batch_size=[4,4] \
        exp_root_dir=$exp_root_dir ${@:6}


python3 launch.py --config configs/nd-mv-nerf/geo-refine.yaml  \
        --train --gpu $GPUS  system.prompt_processor.prompt="$prompt"  use_timestamp=False  \
        name=$geo_refine_out \
        system.geometry_convert_from=$exp_root_dir/$geo_out/$result/ckpts/last.ckpt \
        system.geometry_convert_override.isosurface_threshold=10. \
        trainer.max_steps=1000 exp_root_dir=$exp_root_dir ${@:6}

prompt="a DSLR photo of ${prompt}" # trick proposed By Fantasia3D
python3 launch.py --config configs/nd-mv-nerf/tex.yaml  \
        name=$tex_out \
        system.prompt_processor.prompt="$prompt"  use_timestamp=False  \
        system.geometry_convert_from=$exp_root_dir/$geo_refine_out/$result/ckpts/last.ckpt \
        --train --gpu $GPUS system.prompt_processor.prompt="$prompt"  use_timestamp=False \
        system.nd_guidance_type=$G_TYPE \
        trainer.max_steps=$ITER exp_root_dir=$exp_root_dir ${@:6}


# extract mesh
result=$(echo "${prompt}" | tr ' ' '_')
result=$(echo "$result" | tr -d '"')
python3 launch.py --config $exp_root_dir/$tex_out/$result/configs/parsed.yaml --export --gpu 0 \
    resume=$exp_root_dir/$tex_out/$result/ckpts/last.ckpt system.exporter_type=mesh-exporter \
    system.geometry.isosurface_resolution=256 \
    system.exporter.context_type=cuda exp_root_dir=$exp_root_dir ${@:6}

# copy result
output_path=$exp_root_dir/$tex_out/$result/save/it$ITER-export/*
output_path=$(echo "$output_path" | tr -d '"')
cp -r $output_path $SAVE_MODEL_PATH
