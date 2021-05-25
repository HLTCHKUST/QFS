import argparse
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import Seq2SeqDataset_QFS
from torch.utils.data import DataLoader


logger = getLogger(__name__)

try:
    from .utils import calculate_bleu, calculate_rouge, use_task_specific_params
except ImportError:
    from utils import calculate_bleu, calculate_rouge, use_task_specific_params

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def get_dataset(tokenizer, input_dir, type_path, n_obs) -> Seq2SeqDataset_QFS:
    dataset = Seq2SeqDataset_QFS(
        tokenizer,
        type_path=type_path,
        n_obs=n_obs,
        data_dir=input_dir,
        max_source_length=142,
        max_target_length=48,
    )
    return dataset

def get_dataloader(tokenizer, input_dir, type_path: str, batch_size: int, shuffle: bool = False, n_obs = None) -> DataLoader:
    dataset = get_dataset(tokenizer, input_dir, type_path, n_obs)
    sampler = None
    if type_path == "train":
        sampler = dataset.make_sortish_sampler(batch_size)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle,
        num_workers=8,
        sampler=sampler,
    )
    return dataloader

def generate_summaries_or_translations(
    # examples: List[str],
    data_loader,
    tokenizer,
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    decoder_start_token_id=None,
    **generate_kwargs,
) -> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()

    start_time = time.time()
    # update config with task specific params
    use_task_specific_params(model, task)
    for batch in tqdm(data_loader):
        summaries = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            encoder_answer_relevance_atten=batch['answer_relevance_atten'].to(device),
            # use_cache=True,
            decoder_start_token_id=decoder_start_token_id,
            **generate_kwargs,
        )

        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()

    fout.close()
    runtime = int(time.time() - start_time)  # seconds
    n_obs = len(data_loader)
    return dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4))


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--input_dir", type=str, help="like cnn_dm/")
    parser.add_argument("--save_path", type=str, help="where to save summaries")

    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test_reference_summaries.txt")
    parser.add_argument(
        "--score_path",
        type=str,
        required=False,
        default="metrics.json",
        help="where to save the rouge score in json format",
    )
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument("--task", type=str, default="summarization", help="typically translation or summarization")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument(
        "--decoder_start_token_id",
        type=int,
        default=None,
        required=False,
        help="Defaults to using config",
    )
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name))
    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.


    dataloader = get_dataloader(tokenizer, args.input_dir, type_path = "test", batch_size=args.bs)

    Path(args.save_path).parent.mkdir(exist_ok=True)
    if args.reference_path is None and Path(args.score_path).exists():
        warnings.warn(f"score_path {args.score_path} will be overwritten unless you type ctrl-c.")
    runtime_metrics = generate_summaries_or_translations(
        # examples,
        dataloader,
        tokenizer,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        decoder_start_token_id=args.decoder_start_token_id,
    )
    if args.reference_path is None:
        return
    # Compute scores
    score_fn = calculate_bleu if "translation" in args.task else calculate_rouge
    output_lns = [x.rstrip() for x in open(args.save_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()][: len(output_lns)]
    scores: dict = score_fn(output_lns, reference_lns)
    scores.update(runtime_metrics)
    print(scores)
    if args.score_path is not None:
        json.dump(scores, open(args.score_path, "w"))
    return scores


if __name__ == "__main__":
    # Usage for MT:
    # python run_eval.py MODEL_NAME $DATA_DIR/test.source $save_dir/test_translations.txt --reference_path $DATA_DIR/test.target --score_path $save_dir/test_bleu.json  --task translation $@
    run_generate()
