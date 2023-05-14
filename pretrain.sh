MODELPATH=save_models_reproduce
mkdir -p $MODELPATH
cp *.py $MODELPATH

CUDA_VISIBLE_DEVICES=$1 /gxr/jiahua/anaconda3/envs/omics/bin/python pretrain.py --output_model_dir $MODELPATH --normalize > "$MODELPATH/pretraining.log"
