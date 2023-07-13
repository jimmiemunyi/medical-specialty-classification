from pathlib import Path

from wasabi import Printer
import hydra
from omegaconf import DictConfig, OmegaConf

import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from composer.models import HuggingFaceModel
from composer.metrics import CrossEntropy
from composer import Trainer
from composer.loggers import WandBLogger
from composer.algorithms import GradientClipping

from torchmetrics import F1Score

from utils import (
    create_label_dicts,
    prepare_data,
    prepare_optimizer_and_scheduler,
    predict,
)
from model import MedicalClassification
from awp import AWP

logger = Printer()


@hydra.main(version_base=None, config_path="config", config_name="train")
def train(cfg: DictConfig):
    logger.divider("Setting up")
    print(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

    logger.info("Loading the dataset")
    path = Path(cfg.data.path)

    train_df = pd.read_csv(path / "Train.csv", index_col="Id")
    test_df = pd.read_csv(path / "test.csv", index_col="Id")
    label2int, int2label = create_label_dicts(train_df)
    train_df["labels"] = train_df.medical_specialty.map(label2int)
    num_labels = len(label2int)
    print(f"Number of labels: {num_labels}")

    logger.info("Tokenizing and splitting the dataset")
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.checkpoint)
    train_dl, val_dl = prepare_data(
        train_df, tokenizer, cfg.data.sep_token, cfg.train.batch_size
    )

    logger.info("Loading the model, optimizer, and scheduler")
    f1_score = F1Score(task="multiclass", num_classes=num_labels)
    ptrn_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.train.checkpoint, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    composer_model = HuggingFaceModel(
        model=ptrn_model,
        tokenizer=tokenizer,
        metrics=[CrossEntropy(), f1_score],
        use_logits=True,
        allow_embedding_resizing=True,
    )
    # composer_model = MedicalClassification(
    #     checkpoint=cfg.train.checkpoint,
    #     tokenizer=tokenizer,
    #     pretrained=True,
    #     num_labels=num_labels,
    # )

    # preparing optimizer and scheduler
    optimizer, scheduler = prepare_optimizer_and_scheduler(
        composer_model, cfg.train.lr, cfg.train.wd, cfg.train.epochs, train_dl
    )
    algorithms = []

    if cfg.train.gc:
        gc = GradientClipping(
            clipping_type=cfg.gc.type, clipping_threshold=cfg.gc.value
        )
        algorithms.append(gc)
    if cfg.train.awp:
        awp = AWP(
            start_epoch=cfg.awp.start,
            adv_lr=cfg.awp.adv_lr,
            adv_eps=cfg.awp.adv_eps,
        )
        algorithms.append(awp)

    if cfg.run_name:
        run_name = cfg.run_name
    else:
        run_name = cfg.train.checkpoint.split("/")[-1]

    loggers = []
    if cfg.wandb:
        loggers.append(WandBLogger(project="medical-specialty-classification"))

    logger.info("Loading the trainer")
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dl,
        eval_dataloader=val_dl,
        max_duration=f"{cfg.train.epochs}ep",
        optimizers=optimizer,
        schedulers=[scheduler],
        algorithms=algorithms,
        loggers=loggers,
        run_name=run_name,
        device="gpu",
        precision="amp_fp16",
        step_schedulers_every_batch=True,
        # seed=17,
    )

    logger.divider("Training")
    trainer.fit()
    logger.good(f"Final Validation Scores: {trainer.state.eval_metric_values}")

    if cfg.create_csv:
        logger.divider("Inference")
        test_dl = prepare_data(
            test_df, tokenizer, cfg.data.sep_token, len(test_df), training=False
        )
        preds = predict(trainer, test_dl, int2label)
        logger.info("Creating submission.csv")
        submission = pd.DataFrame({"Id": test_df.index, "medical_specialty": preds})
        submission.to_csv("submission.csv", index=False)
        logger.good("submission.csv created!")


if __name__ == "__main__":
    train()
