for training_type in normal weighted occasional
do
    python train.py --model DLinear --training_type $training_type --verbose
done