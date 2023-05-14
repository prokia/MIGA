

REGRSSION_MODELPATH="runs/regression_finetune"
TRAINED_MODELPATH='save_regression_models'
MODELNAME="model_gnn_5.pth"


mkdir $REGRSSION_MODELPATH
cp src_regression/molecule_finetune_regression.py $REGRSSION_MODELPATH
cp $0 $REGRSSION_MODELPATH


split=scaffold
batch_size=256

CUDA_VISIBLE_DEVICES=0 python src_regression/molecule_finetune_regression.py --dataset esol --batch_size $batch_size --split $split --input_model_file "$TRAINED_MODELPATH/$MODELNAME" --task_type regression > "$REGRSSION_MODELPATH/esol-finetune.log" &
CUDA_VISIBLE_DEVICES=0 python src_regression/molecule_finetune_regression.py --dataset lipo --batch_size $batch_size --split $split --input_model_file "$TRAINED_MODELPATH/$MODELNAME" --task_type regression > "$REGRSSION_MODELPATH/lipo-finetune.log" &
