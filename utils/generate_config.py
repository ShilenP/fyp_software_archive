from utils.config import Config


def get_config_for(data_path: str, pred_len: int, model: None |  str = None, checkpoints_path: str | None = None, trained_models_path: str | None = None, training_type: str = 'normal', n_iter: int = 1, target_epsilon=None, initial_epsilon=None, epsilon_growth_factor=None, initial_kappa=None, kappa_growth_factor=None, final_kappa=None) -> Config:
    """
    Get the configuration for the given dataset path, prediction length and robustness, based off the paper's configurations.
    """
    if model != 'PatchTST':
        config_map ={
            'ETTh1.csv':  Config('ETTh1.csv', pred_len=336, seq_len=336, enc_in=7, d_model=512, seg_len=48, dropout=0.5, train_epochs=30, batch_size=256, learning_rate=0.001, patience=10),
            'ETTh2.csv':  Config('ETTh2.csv', pred_len=336, seq_len=336, enc_in=7, d_model=512, seg_len=48, dropout=0.5, train_epochs=30, batch_size=256, learning_rate=0.0002, patience=10),
            'ETTm1.csv':  Config('ETTm1.csv', pred_len=336, seq_len=336, enc_in=7, d_model=512, seg_len=48, dropout=0.5, train_epochs=30, batch_size=256, learning_rate=0.0002, patience=10),
            'ETTm2.csv':  Config('ETTm2.csv', pred_len=336, seq_len=336, enc_in=7, d_model=512, seg_len=48, dropout=0.5, train_epochs=30, batch_size=256, learning_rate=0.0001, patience=10),
            'electricity.csv':  Config('electricity.csv', pred_len=336, seq_len=336, enc_in=321, d_model=512, seg_len=48, dropout=0.1, train_epochs=30, batch_size=16, learning_rate=0.0005, patience=10),
            'traffic.csv':  Config('traffic.csv', pred_len=336, seq_len=336, enc_in=862, d_model=512, seg_len=48, dropout=0.1, train_epochs=30, batch_size=8, learning_rate=0.003, patience=10),
            'weather.csv':  Config('weather.csv', pred_len=336, seq_len=336, enc_in=21, d_model=512, seg_len=48, dropout=0.5, train_epochs=30, batch_size=64, learning_rate=0.0001, patience=10)
        }
    else:
        config_map ={
            'ETTh1.csv':  Config('ETTh1.csv', pred_len=336, seq_len=336, enc_in=7, e_layers=3, n_heads=4, d_model=16, d_ff=128, dropout=0.3, fc_dropout=0.3, head_dropout=0, train_epochs=100, patch_len=16, stride=8, batch_size=128, learning_rate=0.0001),
            'ETTh2.csv':  Config('ETTh2.csv', pred_len=336, seq_len=336, enc_in=7, e_layers=3, n_heads=4, d_model=16, d_ff=128, dropout=0.3, fc_dropout=0.3, head_dropout=0, train_epochs=100, patch_len=16, stride=8, batch_size=128, learning_rate=0.0001),
            'ETTm1.csv':  Config('ETTm1.csv', pred_len=336, seq_len=336, enc_in=7, e_layers=3, n_heads=16, d_model=128, d_ff=256, dropout=0.2, fc_dropout=0.2, head_dropout=0, train_epochs=100, patch_len=16, stride=8, batch_size=128, learning_rate=0.0001, patience=20, pct_start=0.4, lradj='TST'),
            'ETTm2.csv':  Config('ETTm2.csv', pred_len=336, seq_len=336, enc_in=7, e_layers=3, n_heads=16, d_model=128, d_ff=256, dropout=0.2, fc_dropout=0.2, head_dropout=0, train_epochs=100, patch_len=16, stride=8, batch_size=128, learning_rate=0.0001, patience=20, pct_start=0.4, lradj='TST'),
            'electricity.csv':  Config('electricity.csv', pred_len=336, seq_len=336, enc_in=321, e_layers=3, n_heads=16, d_model=128, d_ff=256, dropout=0.2, fc_dropout=0.2, head_dropout=0, train_epochs=100, patch_len=16, stride=8, batch_size=32, learning_rate=0.0001, patience=10, pct_start=0.2, lradj='TST'),
            'traffic.csv':  Config('traffic.csv', pred_len=336, seq_len=336, enc_in=862, e_layers=3, n_heads=16, d_model=128, d_ff=256, dropout=0.2, fc_dropout=0.2, head_dropout=0, train_epochs=100, patch_len=16, stride=8, batch_size=24, learning_rate=0.0001, patience=10, pct_start=0.2, lradj='TST'),
            'weather.csv':  Config('weather.csv', pred_len=336, seq_len=336, enc_in=21, e_layers=3, n_heads=16, d_model=128, d_ff=256, dropout=0.2, fc_dropout=0.2, head_dropout=0, train_epochs=100, patch_len=16, stride=8, batch_size=128, learning_rate=0.0001, patience=20),
        }
    config = config_map[data_path]
    config.training_type = training_type
    config.n_iter = n_iter
    config.pred_len = pred_len
    if checkpoints_path is not None:
        config.checkpoints_path = checkpoints_path
    if trained_models_path is not None:
        config.trained_models_path = trained_models_path

    if target_epsilon is not None:
        config.target_epsilon = target_epsilon
    if initial_epsilon is not None:
        config.initial_epsilon = initial_epsilon
    if epsilon_growth_factor is not None:
        config.epsilon_growth_factor = epsilon_growth_factor
    if initial_kappa is not None:
        config.initial_kappa = initial_kappa
    if kappa_growth_factor is not None:
        config.kappa_growth_factor = kappa_growth_factor
    if final_kappa is not None:
        config.final_kappa = final_kappa

    return config
