MODELPATH=save_models_mgm
mkdir -p $MODELPATH
cp *.py $MODELPATH

CUDA_VISIBLE_DEVICES=$1 /gxr/jiahua/anaconda3/envs/omics/bin/python pretrain.py --output_model_dir $MODELPATH --MGM_mode MGM --normalize > "$MODELPATH/pretraining.log"