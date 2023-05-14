#### Clinical fine-tuning
MODELPATH="./runs/results_clinical"
TRAINED_MODELPATH='./models_MGM_training_resnet34_image10'
MODELNAME="model_gnn_85.pth"

mkdir $MODELPATH
cp src_clinical/clinical_finetune.py $MODELPATH
cp $0 $MODELPATH

split=defined
batch_size=64

CUDA_VISIBLE_DEVICES=$1 python src_clinical/clinical_finetune.py --dataset phaseI --batch_size $batch_size --epochs 100 --split $split --input_model_file "$TRAINED_MODELPATH/$MODELNAME" > "$MODELPATH/phaseI-finetune.log" &
CUDA_VISIBLE_DEVICES=$1 python src_clinical/clinical_finetune.py --dataset phaseII --batch_size $batch_size --epochs 100 --split $split --input_model_file "$TRAINED_MODELPATH/$MODELNAME" > "$MODELPATH/phaseII-finetune.log" &
CUDA_VISIBLE_DEVICES=$1 python src_clinical/clinical_finetune.py --dataset phaseIII --batch_size $batch_size --epochs 100 --split $split --input_model_file "$TRAINED_MODELPATH/$MODELNAME" > "$MODELPATH/phaseIII-finetune.log" &
