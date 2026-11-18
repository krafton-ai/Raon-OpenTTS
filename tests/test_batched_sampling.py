"""Batched inference gives each row what it gives alone; training is left as it was.

`CFM.sample` pads the rows of a batch to the longest target duration. At inference the DiT masks that padding in
the attention, the conv position embedding and the text encoder's GRN, so a shorter row does not see it. The checks
use a small DiT with random weights and the options of the released configs, on CPU in fp32 (a few seconds).
"""

import torch
from torch.nn.utils.rnn import pad_sequence

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.cfm import CFM
from f5_tts.model.utils import lens_to_mask, list_str_to_idx

VOCAB = {c: i for i, c in enumerate(" abcdefghijklmnopqrstuvwxyz.,'")}

# (reference frames, total duration, text); the first row is the longest, so it is never padded
ROWS = [
    (23, 63, "the cat sat on the mat. it was a sunny day"),
    (9, 27, "hello there. how are you"),
    (15, 44, "one two three. four five six seven"),
]


def tiny_model():
    torch.manual_seed(1234)
    dit = DiT(
        dim=64,
        depth=3,
        heads=4,
        dim_head=16,
        ff_mult=2,
        mel_dim=80,
        text_num_embeds=len(VOCAB),
        text_dim=32,
        text_mask_padding=True,
        conv_layers=2,
        attn_mask_enabled=False,
        dropout=0.0,
        norm_type="rmsnorm",
    )
    model = CFM(transformer=dit, mel_spec_kwargs={"n_mel_channels": 80}, vocab_char_map=VOCAB)
    # random weights: the DiT zero-inits its AdaLN and output layers, which would make every output 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    return model


def make_rows():
    gen = torch.Generator().manual_seed(7)
    return [(torch.randn(n_ref, 80, generator=gen), dur, text) for n_ref, dur, text in ROWS]


def sample(model, rows, steps=4):
    """Each row's output (its own duration) from one `sample` call over all rows, one seed as in released usage."""
    cond = pad_sequence([mel for mel, _, _ in rows], batch_first=True)
    lens = torch.tensor([mel.shape[0] for mel, _, _ in rows])
    duration = torch.tensor([dur for _, dur, _ in rows])
    out, _ = model.sample(
        cond=cond,
        text=[text for _, _, text in rows],
        duration=duration,
        lens=lens,
        steps=steps,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        seed=0,
    )
    return [out[i, :dur] for i, (_, dur, _) in enumerate(rows)]


def test_padded_batch_matches_single_inference():
    model = tiny_model()
    rows = make_rows()
    batched = sample(model, rows)
    for row, out in zip(rows, batched):
        alone = sample(model, [row])[0]
        # batched kernels round differently from single ones: every row, the never-padded first one included, moves
        # by about 2e-6 here. with the padding visible the shorter rows move by about 0.6, and by 2e-3 or more if any
        # one of the three paths (attention, conv position embedding, GRN) is left unmasked
        torch.testing.assert_close(out, alone, rtol=0, atol=1e-4)


def test_training_forward_does_not_mask_padding():
    model = tiny_model()
    dit = model.transformer
    rows = make_rows()
    n = max(dur for _, dur, _ in rows)
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(len(rows), n, 80, generator=gen)
    cond = torch.randn(len(rows), n, 80, generator=gen)
    text = list_str_to_idx([text for _, _, text in rows], VOCAB)
    time = torch.rand(len(rows), generator=gen)
    mask = lens_to_mask(torch.tensor([dur for _, dur, _ in rows]))

    def forward(mask):
        with torch.no_grad():
            return dit(x=x, cond=cond, text=text, time=time, mask=mask)

    # the training forward as it always was: the embeddings never see the mask, and the attention (attn_mask_enabled
    # False) only zeroes its output at the padding. here: no mask at all, and that zeroing done by a hook
    dit.eval()
    hooks = [
        block.attn.register_forward_hook(lambda module, args, out: out.masked_fill(~mask[..., None], 0.0))
        for block in dit.transformer_blocks
    ]
    reference = forward(None)
    for hook in hooks:
        hook.remove()

    inference = forward(mask)
    dit.train()
    training = forward(mask)

    assert torch.equal(training, reference)
    # the same call at inference does mask the padding, which moves the shorter rows
    assert (inference - reference).abs().amax() > 1e-2
