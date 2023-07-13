"""Some Utility Functions that help iterate faster."""
from collections import Counter
from functools import partial

import pandas as pd
import numpy as np

from torch import tensor
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.data.data_collator import default_data_collator

from composer import Trainer
from composer.models import HuggingFaceModel
from composer.metrics import CrossEntropy
from composer.loggers import WandBLogger

from torchmetrics import F1Score


def process_df(df: pd.DataFrame, sep_token: str):
    df["input"] = df.sample_name + sep_token + df.description + sep_token
    df.input = df.input.str.lower()
    return df


def create_val_split(df: pd.DataFrame, val_prop: float = 0.2, seed: int = 42):
    sample_name = df.sample_name.unique()
    np.random.seed(seed)
    np.random.shuffle(sample_name)
    val_sz = int(len(sample_name) * val_prop)
    val_sample_name = sample_name[:val_sz]
    is_val = np.isin(df.sample_name, val_sample_name)
    idxs = np.arange(len(df))
    val_idxs = idxs[is_val]
    trn_idxs = idxs[~is_val]

    return trn_idxs, val_idxs


def tokenize_func(batch, tokenizer):
    return tokenizer(
        batch["input"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


def create_label_dicts(df: pd.DataFrame):
    label2int = {label: i for i, label in enumerate(df.medical_specialty.unique())}
    int2label = {v: k for k, v in label2int.items()}

    return label2int, int2label


def tokenize_and_split(df, tokenize_func, train=True):
    remove_columns = ["sample_name", "input", "transcription"]
    dataset = datasets.Dataset.from_pandas(df)
    tok_dataset = dataset.map(
        tokenize_func, batched=True, batch_size=None, remove_columns=remove_columns
    )
    if train:
        # remove_columns.append("medical_specialty")
        trn_idxs, val_idxs = create_val_split(df)
        tok_dataset = datasets.DatasetDict(
            {
                "train": tok_dataset.select(trn_idxs),
                "test": tok_dataset.select(val_idxs),
            }
        )

    return tok_dataset


def make_balanced_sampler(tok_ds):
    labels = tok_ds["train"]["labels"]
    label_counts = Counter(labels)
    weights = 1 / np.array([label_counts[label] for label in labels])
    weights = tensor(weights)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
    )
    return sampler


def create_dataloaders(tok_ds, bs, sampler=None, train=True):
    if train:
        train_dl = DataLoader(
            tok_ds["train"],
            batch_size=bs,
            # shuffle=True,
            collate_fn=default_data_collator,
            drop_last=True,
            sampler=sampler,
        )
        val_dl = DataLoader(
            tok_ds["test"],
            batch_size=bs,
            shuffle=False,
            collate_fn=default_data_collator,
            drop_last=True,
        )

        return train_dl, val_dl
    else:
        test_dl = DataLoader(
            tok_ds,
            batch_size=bs,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        return test_dl


def prepare_data(df, tokenizer, sep_token, bs, training=True):
    if training:
        train_df = process_df(df, sep_token)
        tokenize = partial(tokenize_func, tokenizer=tokenizer)
        train_tok_ds = tokenize_and_split(train_df, tokenize)
        sampler = make_balanced_sampler(train_tok_ds)
        train_dl, val_dl = create_dataloaders(train_tok_ds, bs, sampler)

        return train_dl, val_dl
    else:
        test_df = process_df(df, sep_token)
        tokenize = partial(tokenize_func, tokenizer=tokenizer)
        test_tok_ds = tokenize_and_split(test_df, tokenize, train=False)
        test_dl = create_dataloaders(test_tok_ds, bs, train=False)

        return test_dl


def prepare_optimizer_and_scheduler(model, lr, wd, epochs, train_dl):
    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=wd,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_dl),
        epochs=epochs,
    )

    return optimizer, scheduler


def prepare_model(checkpoint, tokenizer, num_labels):
    f1_score = F1Score(task="multiclass", num_classes=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
    )
    hf_model = HuggingFaceModel(
        model, tokenizer=tokenizer, metrics=[CrossEntropy(), f1_score], use_logits=True
    )

    return hf_model


def prepare_trainer(
    composer_model, optimizer, scheduler, train_dl, val_dl, epochs, run_name
):
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dl,
        eval_dataloader=val_dl,
        max_duration=f"{epochs}ep",
        optimizers=optimizer,
        schedulers=[scheduler],
        loggers=[WandBLogger(project="medical-specialty-classification")],
        run_name=run_name,
        device="gpu",
        precision="amp_fp16",
        step_schedulers_every_batch=True,
        # seed=17,
    )

    return trainer


def fit(
    train_df,
    cfg,
    run_name,
    num_labels,
):
    # preparing data
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.checkpoint)
    train_dl, val_dl = prepare_data(
        train_df, tokenizer, cfg.train.sep_token, cfg.train.batch_size
    )
    model = prepare_model(cfg.train.checkpoint, tokenizer, num_labels)

    # preparing optimizer and scheduler
    optimizer, scheduler = prepare_optimizer_and_scheduler(
        model, cfg.train.lr, cfg.train.wd, cfg.train.epochs, train_dl
    )

    # preparing trainer
    trainer = prepare_trainer(
        model, optimizer, scheduler, train_dl, val_dl, cfg.train.epochs, run_name
    )

    # training
    trainer.fit()

    return trainer


def predict(trainer, test_dl, int2label, custom=False):
    # implement custom predict function
    preds = trainer.predict(test_dl)[0]["logits"].argmax(1).numpy().astype(int)
    preds = preds.tolist()
    preds = [int2label[pred] for pred in preds]
    return preds
