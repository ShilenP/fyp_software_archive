for training_type in normal weighted occasional
do
    python train.py --model PatchTST --training_type $training_type --verbose
done