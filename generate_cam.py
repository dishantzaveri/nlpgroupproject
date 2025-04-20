#!/usr/bin/env python
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

def parse_args():
    parser = ArgumentParser(
        prog='generate',
        description='Run LLaVA on MileBench, optionally with Cache‑Merging (CaM)'
    )
    parser.add_argument('--data_dir',        default='data/MileBench')
    parser.add_argument('--dataset_name',    default='sample')  
    parser.add_argument('--model_name',      required=True)
    parser.add_argument('--output_dir',      default='outputs')
    parser.add_argument('--bsz',             default=1, type=int)
    parser.add_argument('--batch-image',     default=1, type=int,
                        help='Images per batch divisor')
    parser.add_argument('--combine_image',   default=None, type=int,
                        help='Combine N images into one prompt')
    parser.add_argument('--model_configs',   default='configs/model_configs.yaml')
    parser.add_argument('--overwrite',       action='store_true')
    parser.add_argument('--enable_cam',       action='store_true',
                        help='Enable Cache‑Merging on LLaMA attention')
    parser.add_argument('--cam_start_ratio',  type=float, default=0.1,
                        help='Fraction of oldest KV cache to keep')
    parser.add_argument('--cam_recent_ratio', type=float, default=0.1,
                        help='Fraction of most recent KV cache to keep')

    args = parser.parse_args()
    args.output_pth = os.path.join(
        args.output_dir,
        args.model_name,
        args.dataset_name,
        'predictions_no_cam.json'
    )
    os.makedirs(os.path.dirname(args.output_pth), exist_ok=True)
    return args

def split_data(data):
    data_dict = {}
    for d in data:
        n_img = len(d['task_instance']['images_path'])
        data_dict.setdefault(n_img, []).append(d)
    return data_dict

def main(args):
    print(f"{datetime.now()}: Inference with {args.model_name} on {args.dataset_name}")

    # Load annotation JSON
    ds_dir  = os.path.join(args.data_dir, args.dataset_name)
    img_dir = os.path.join(ds_dir, 'images')
    fn = (f"{args.dataset_name}_combined_{args.combine_image}.json"
          if args.combine_image
          else f"{args.dataset_name}.json")
    core = json.load(open(os.path.join(ds_dir, fn), 'r'))

    # Partition by image count
    data_dict = split_data(core['data'])

    # Build worker
    WorkerCls   = get_worker_class(args.model_name)
    all_configs = OmegaConf.load(args.model_configs)
    config      = all_configs[args.model_name]
    config.device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.enable_cam   = args.enable_cam
    config.start_ratio  = args.cam_start_ratio
    config.recent_ratio = args.cam_recent_ratio

    worker = WorkerCls.from_config(config=config)
    worker.model.to(config.device)
    worker.model.eval()

    # Stats collectors
    mem_stats     = []
    latency_stats = []
    total_samples = 0

    results = []
    for n_img, samples in data_dict.items():
        print(f"→ {len(samples)} examples with {n_img} images each")
        ds = MileBenchDataset(
            annotation          = samples,
            task_instructions   = core['meta_data']['task_instruction'],
            img_dir             = img_dir,
            max_context_len     = config.max_context_len,
            n_tokens_per_image  = config.n_tokens_per_image,
            tokenizer           = worker.tokenizer,
            dataset_name        = args.dataset_name,
            combine_image       = args.combine_image,
        )
        loader = DataLoader(
            ds,
            batch_size  = max(args.batch_image // n_img, 1),
            shuffle     = False,
            num_workers = 0,
            collate_fn  = ds.collate_fn
        )

        for batch in tqdm(loader, desc=f"{n_img}-img batches"):
            # reset mem stats
            if config.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(config.device)
    
            t0 = time.perf_counter()
            with torch.no_grad():
                outs = worker(device=config.device, **batch)
            dt = time.perf_counter() - t0
    
            # derive batch_size from the number of outputs
            batch_size = len(outs)
    
            if config.device.startswith('cuda'):
                peak_mem = torch.cuda.max_memory_allocated(config.device) / 2**30
            else:
                peak_mem = 0.0
    
            latency_stats.append(dt / batch_size * 1000)  # ms per sample
            mem_stats.append(peak_mem)
    
            results.extend(outs)
            
    # Save predictions
    with open(args.output_pth, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved → {args.output_pth}\n")

    # Summarize metrics
    print("=== Performance Metrics ===")
    print(f"Total samples:       {total_samples}")
    print(f"Avg latency:         {np.mean(latency_stats):.1f} ms/sample  "
          f"(±{np.std(latency_stats):.1f})")
    if config.device.startswith('cuda'):
        print(f"Peak GPU Memory:     {np.mean(mem_stats):.2f} GB  "
              f"(±{np.std(mem_stats):.2f})")
    print("===========================")

if __name__ == '__main__':
    args = parse_args()
    main(args)
