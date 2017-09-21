#!/usr/bin/env bash
python -m nmt.nmt \
    --src=articles --tgt=abstracts \
    --hparams_path=textsumdata/textsum.json \
    --out_dir=out_models/ \
    --vocab_prefix=textsumdata/vocab \
    --train_prefix=textsumdata/train \
    --dev_prefix=textsumdata/dev \
    --test_prefix=textsumdata/test \
    --num_gpus=0