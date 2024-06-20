import os
class Config:
    def __init__(self, data_path='exchange_rate.csv', pred_len=96, seq_len=336, label_len=48, batch_size=32, learning_rate=0.0001, enc_in=7, dec_in=None, c_out=None, factor=1, train_epochs=10, patience=100, dropout=0.05, fc_dropout=0.05, head_dropout=0, patch_len=16, stride=8, d_ff=2048, d_model=512, n_heads=8, e_layers=2, seg_len=48, lradj='type1', pct_start=0.3, training_type='normal', n_iter=1, target_epsilon=0.01, initial_epsilon=0, epsilon_growth_factor=2, initial_kappa=0, kappa_growth_factor=2, final_kappa=0.5):
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.label_len = label_len
        self.freq = 'h'
        self.batch_size = batch_size
        self.root_path = './dataset/'
        self.data_path = data_path
        self.features = 'M'
        self.target = 'OT'
        self.num_workers = 6
        self.embed = 'timeF'
        self.train_epochs = train_epochs
        self.device = 'cuda:0'
        self.patience = patience
        self.learning_rate = learning_rate
        self.checkpoints_path = './checkpoints/'
        self.lradj = lradj

        self.individual = False
        self.enc_in = enc_in

        self.trained_models_path = './trained_models/'

        # Varied params for Autoformer
        self.dec_in = enc_in if dec_in is None else dec_in
        self.c_out = enc_in if c_out is None else c_out
        self.factor = factor
        self.dropout = dropout
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff

        # Fixed params for Autoformer
        self.moving_avg = 25
        self.activation = 'gelu'
        self.d_layers = 1

        # Varied params for PatchTST
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.patch_len = patch_len
        self.stride = stride
        self.pct_start = pct_start

        # SegRNN
        self.seg_len = seg_len # segment length

        # Training type
        self.training_type = training_type # 'normal', 'occasional' or 'weighted'

        # Used for the occasional robust loss training
        self.n_iter = n_iter

        # Used for the weighted robust loss training
        self.target_epsilon = target_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_growth_factor = epsilon_growth_factor
        self.initial_kappa = initial_kappa
        self.kappa_growth_factor = kappa_growth_factor
        self.final_kappa = final_kappa

    def _get_training_type(self):
        if self.training_type == 'occasional':
            return os.path.join(self.training_type, str(self.n_iter))
        return self.training_type

    def get_checkpoint_path(self, model):
        return os.path.join(self.checkpoints_path, str(model), self.data_path.split('.')[0], str(self.pred_len), self._get_training_type())

    def get_trained_model_path(self, model):
        return os.path.join(self.trained_models_path, str(model), self.data_path.split('.')[0], str(self.pred_len), self._get_training_type())

    def __str__(self):
        return '&'.join(f'{self.__dict__[key]}' for key in sorted(self.__dict__) if 'path' not in key)