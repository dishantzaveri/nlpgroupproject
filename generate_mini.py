import os
import time
import json
import torch
import numpy as np

from datetime import datetime
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from utils import get_worker_class, MileBenchDataset

from workers.minigpt4_worker import MiniGPT4Worker 

def main(args):
    print(f"{datetime.now()}: Inference with {args.model_name} on {args.dataset_name}")

    ds_dir  = os.path.join(args.data_dir, args.dataset_name)
    img_dir = os.path.join(ds_dir, 'images')
    fn = (f"{args.dataset_name}_combined_{args.combine_image}.json"
          if args.combine_image
          else f"{args.dataset_name}.json")
    core = json.load(open(os.path.join(ds_dir, fn), 'r'))

    data_dict = split_data(core['data'])

    all_configs = OmegaConf.load(args.model_configs)
    config = all_configs[args.model_name]
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.enable_cam = args.enable_cam
    config.start_ratio = args.cam_start_ratio
    config.recent_ratio = args.cam_recent_ratio

    worker = MiniGPT4Worker.from_config(config=config)
    worker.model.to(config.device)
    worker.model.eval()

    mem_stats = []
    latency_stats = []
    total_samples = 0
    results = []

    for n_img, samples in data_dict.items():
        print(f"→ {len(samples)} examples with {n_img} images each")
        ds = MileBenchDataset(
            annotation=samples,
            task_instructions=core['meta_data']['task_instruction'],
            img_dir=img_dir,
            max_context_len=config.max_context_len,
            n_tokens_per_image=config.n_tokens_per_image,
            tokenizer=worker.tokenizer,
            dataset_name=args.dataset_name,
            combine_image=args.combine_image,
        )
        loader = DataLoader(
            ds,
            batch_size=max(args.batch_image // n_img, 1),
            shuffle=False,
            num_workers=0,
            collate_fn=ds.collate_fn
        )

        for batch in tqdm(loader, desc=f"{n_img}-img batches"):
            if config.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(config.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                outs = worker(device=config.device, **batch)
            dt = time.perf_counter() - t0

            batch_size = len(outs)
            total_samples += batch_size

            if config.device.startswith('cuda'):
                peak_mem = torch.cuda.max_memory_allocated(config.device) / 2**30
            else:
                peak_mem = 0.0

            latency_stats.append(dt / batch_size * 1000)
            mem_stats.append(peak_mem)
            results.extend(outs)

    with open(args.output_pth, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved → {args.output_pth}\n")

    print("=== Performance Metrics ===")
    print(f"Total samples:       {total_samples}")
    print(f"Avg latency:         {np.mean(latency_stats):.1f} ms/sample  "
          f"(±{np.std(latency_stats):.1f})")
    if config.device.startswith('cuda'):
        print(f"Peak GPU Memory:     {np.mean(mem_stats):.2f} GB  "
              f"(±{np.std(mem_stats):.2f})")
    print("===========================")
