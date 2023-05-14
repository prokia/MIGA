


MODELPATH="./runs/classification_finetune"
TRAINED_MODELPATH='./models_MGM_training_resnet34_image10'
MODELNAME="model_gnn_85.pth"


mkdir $MODELPATH
cp src_classification/molecule_finetune_classification.py $MODELPATH
cp $0 $MODELPATH


split=scaffold
batch_size=512

CUDA_VISIBLE_DEVICES=$1 python src_classification/molecule_finetune_classification.py --dataset bbbp --batch_size $batch_size --epochs 100 --split $split --input_model_file "$TRAINED_MODELPATH/$MODELNAME" > "$MODELPATH/bbbp-finetune.log" &
CUDA_VISIBLE_DEVICES=$1 python src_classification/molecule_finetune_classification.py --dataset tox21 --batch_size $batch_size --epochs 100 --split $split --input_model_file "$TRAINED_MODELPATH/$MODELNAME" > "$MODELPATH/tox21-finetune.log" &
CUDA_VISIBLE_DEVICES=$1 python src_classification/molecule_finetune_classification.py --dataset toxcast --batch_size $batch_size --epochs 100 --split $split --input_model_file "$TRAINED_MODELPATH/$MODELNAME" > "$MODELPATH/toxcast-finetune.log" &
CUDA_VISIBLE_DEVICES=$1 python src_classification/molecule_finetune_classification.py --dataset hiv --batch_size $batch_size --epochs 100 --split $split --input_model_file "$TRAINED_MODELPATH/$MODELNAME" > "$MODELPATH/hiv-finetune.log" &
