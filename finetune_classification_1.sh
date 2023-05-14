


MODELPATH="./run_classification_finetune"


mkdir $MODELPATH

CUDA_VISIBLE_DEVICES=4 python src_classification/molecule_finetune_classification.py  --dataset tox21 --batch_size 55 --epochs 150 --input_model_file /rhome/lianyu.zhou/cache/MIGA/miga_transformer_b128_l3/2022-12-15-03-51-46/weight/model_gnn450.pth --split scaffold --pretrain_cfg_path config/miga/miga_graphTrans.yaml --pretrain_model_path /rhome/lianyu.zhou/cache/MIGA/miga_transformer_b128_l3_atom60/2023-2-4-15-26-18/weight/model_700.pth --num_run 100 --lr 3e-4 --seed 123 --runseed 1 --decay 1e-5 | tee > "$MODELPATH/tox21-finetune.log"
