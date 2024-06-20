for training_type in normal weighted occasional
do
    python train.py --model SegRNN --training_type $training_type --verbose
done