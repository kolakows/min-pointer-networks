import torch
import torch.nn as nn
import torch.nn.functional as F


class PtrNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, start_token_value=-1.1):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.w1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Parameter(torch.randn(hidden_size))
        self.register_buffer('start_token', torch.tensor(start_token_value))
        self.input_size = input_size

    def forward(self, X, y=None):
        bs, seq, _ = X.shape
        hidden, encoder_state = self.encoder(X) # shape of hidden (bs, seq, h)
        e_proj = self.w1(hidden) # shape of e_proj (bs, seq, h)

        init_seq = self.start_token.expand(bs, 1, self.input_size)
        _, last_state = self.decoder(init_seq, encoder_state) # h_n shape is (bs, 1, h)

        all_logits = torch.zeros((bs, seq, seq), device=X.device) 
        indices = torch.zeros((bs, seq), dtype=torch.long, device=X.device)
        mask = torch.zeros((bs, seq), dtype=torch.bool, device=X.device)

        for i in range(X.shape[1]):
            h, _ = last_state
            h_proj = self.w2(h)
            h_proj = h_proj.squeeze(0).unsqueeze(1) # more efficient rearrange(h_proj, "1 b d -> b 1 d")

            logits = F.tanh(e_proj + h_proj) @ self.v # (bs, seq, h) + (bs, 1, h), broadcasts automatically
            all_logits[:, i, :] = logits # (bs, seq)
            # masking already chosen indices
            logits[mask] = -1e9

            probs = F.softmax(logits, dim=1)
            chosen = torch.argmax(logits, dim=1) # (bs)
            indices[:, i] = chosen
            
            mask[torch.arange(bs), chosen] = True

            if y is None:
                chosen_input = X[torch.arange(bs), chosen, None]
            else:
                # teacher forcing
                chosen_input = X[torch.arange(bs), y[:, i], None]

            _, last_state = self.decoder(chosen_input, last_state)

        if y is not None:
            loss = F.cross_entropy(all_logits.view(-1, seq), y.view(-1), reduction='mean')
            return indices, all_logits, loss
        else:
            return indices, all_logits