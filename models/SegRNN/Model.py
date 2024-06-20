# SegRNN model adpated from the original implementation in the paper "SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting" by Shengsheng Lin, Weiwei Lin, Wentai Wu, Feiyu Zhao, Ruichao Mo, Haotong Zhang. https://github.com/lss-1138/SegRNN/blob/main/models/SegRNN.py

import torch
import torch.nn as nn
from fastbound.bound_propagation import ibp_propagate_gru, ibp_propagate_linear, ibp_propagation
from utils.config import Config

class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()

        self.seq_len = config.seq_len
        self.seg_len = config.seg_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        self.d_model = config.d_model
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        self.valueEmbedding = nn.Sequential(
            nn.Linear(config.seg_len, config.d_model),
            nn.ReLU()
        )

        # self.rnn = GRULayer(input_size=config.d_model, hidden_size=config.d_model)
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                        batch_first=True, bidirectional=False)

        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, config.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(config.enc_in, config.d_model // 2))

        self.predict = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.seg_len)
        )

    def forward_bounds(self, lower_bound, upper_bound):
        batch_size = lower_bound.size(0)

        lower_seq_last, upper_seq_last = (lower_bound[:, -1:, :].detach()), (upper_bound[:, -1:, :].detach())
        lower_bound, upper_bound = (lower_bound - upper_seq_last).permute(0, 2, 1), (upper_bound - lower_seq_last).permute(0, 2, 1)

        lower_bound, upper_bound = (lower_bound.reshape(-1, self.seg_num_x, self.seg_len)), (upper_bound.reshape(-1, self.seg_num_x, self.seg_len))

        lower_bound, upper_bound = ibp_propagation(lower_bound, upper_bound, self.valueEmbedding[0])
        lower_bound, upper_bound = ibp_propagation(lower_bound, upper_bound, self.valueEmbedding[1])

        hidden_lower, hidden_upper = torch.zeros(lower_bound.size(0), self.d_model, device=lower_bound.device), torch.zeros(upper_bound.size(0), self.d_model, device=upper_bound.device)
        _, _, hn_lower, hn_upper = ibp_propagate_gru(lower_bound, upper_bound, hidden_lower, hidden_upper, self.rnn)
        pos_emb_repeated = self.pos_emb.unsqueeze(0) + torch.zeros(self.enc_in, self.seg_num_y, self.d_model //2, device=lower_bound.device)
        channel_emb_repeated = self.channel_emb.unsqueeze(1) + torch.zeros(self.enc_in, self.seg_num_y, self.d_model //2, device=lower_bound.device)

        pos_emb = torch.cat([
            pos_emb_repeated,
            channel_emb_repeated
        ], dim=-1).view(1, -1, 1, self.d_model) + torch.zeros(batch_size, self.enc_in * self.seg_num_y , 1, self.d_model, device=lower_bound.device)

        pos_emb = pos_emb.reshape(-1, 1, self.d_model)

        hn_lower = hn_lower.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)
        hn_upper = hn_upper.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)

        _, _, hy_lower, hy_upper = ibp_propagate_gru(pos_emb, pos_emb, hn_lower, hn_upper, self.rnn)

        y_lower, y_upper = ibp_propagate_linear(hy_lower, hy_upper, self.predict[1].weight, self.predict[1].bias)
        y_lower = y_lower.view(-1, self.enc_in, self.pred_len)
        y_upper = y_upper.view(-1, self.enc_in, self.pred_len)

        y_lower = y_lower.permute(0, 2, 1) + lower_seq_last
        y_upper = y_upper.permute(0, 2, 1) + upper_seq_last
        
        return y_lower, y_upper

    def forward(self, x: torch.Tensor):
        """
        x: input tensor of shape (batch_size, seq_len, enc_in)
        """

        batch_size = x.size(0)
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1) # (batch_size, enc_in, seq_len)

        x = x.reshape(-1, self.seg_num_x, self.seg_len) # (batch_size * enc_in, seg_num_x, seg_len)
        x = self.valueEmbedding(x) # (batch_size * enc_in, seg_num_x, d_model)
        _, hn = self.rnn(x) # (batch_size * enc_in, seg_num_x, d_model)


        pos_emb_repeated = self.pos_emb.unsqueeze(0) + torch.zeros(self.enc_in, self.seg_num_y, self.d_model //2, device=x.device)
        channel_emb_repeated = self.channel_emb.unsqueeze(1) + torch.zeros(self.enc_in, self.seg_num_y, self.d_model //2, device=x.device)

        pos_emb = torch.cat([
            pos_emb_repeated,
            channel_emb_repeated
        ], dim=-1).view(1, -1, 1, self.d_model) + torch.zeros(batch_size, self.enc_in * self.seg_num_y , 1, self.d_model, device=x.device)

        pos_emb = pos_emb.reshape(-1, 1, self.d_model)

        hn = hn.squeeze(0)
        hn_expanded = ((hn.unsqueeze(-1) + torch.zeros(batch_size * self.enc_in, self.d_model, self.seg_num_y, device=x.device)).permute(0, 2, 1).reshape(1, batch_size * self.enc_in * self.seg_num_y, self.d_model))
        _, hy = self.rnn(pos_emb, hn_expanded)
        y  = self.predict(hy) # (batch_size * enc_in, seg_num_y, seg_len)

        y = y.view(-1, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) + seq_last
        return y
    
    def __str__(self):
        return 'SegRNN'
