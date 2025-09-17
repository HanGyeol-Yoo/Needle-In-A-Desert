#!/usr/bin/env python3
import os
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def visualize_heatmap(df_raw, save_path: str, model_name: str):
    df = df_raw.copy()
    if df.empty:
        print(f"[WARN] Empty DataFrame for {model_name}, skipping visualization.")
        return

    df['Context Length'] = df['dataset'].apply(lambda x: int(re.search(r'Length(\d+)', x).group(1)))
    df['Document Depth'] = df['dataset'].apply(lambda x: float(re.search(r'Depth(\d+)', x).group(1)))

    pivot = pd.pivot_table(df, values=model_name, index='Document Depth', columns='Context Length')
    overall_score = pivot.mean().mean()

    plt.figure(figsize=(8.5, 4.5))
    ax = plt.gca()
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#F0496E', '#EBB839', '#0CD79F'])
    sns.heatmap(pivot, cmap=cmap, ax=ax, vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'})

    ax2 = ax.twinx()
    mean_scores = pivot.mean().values
    
    x_data = [i + 0.5 for i in range(len(mean_scores))] 
    
    ax2.plot(x_data, mean_scores, color='white', marker='o', linestyle='-', linewidth=2, markersize=8, label='Average Score per Length')
    ax2.set_ylim(0, 100)
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    ax2.legend(loc='lower left')


    for i, score in enumerate(mean_scores):
        ax2.text(x_data[i], score - 1, f'{score:.2f}', ha='center', va='top', color='black', fontsize=7, fontweight='bold')

    ax.set_title(f"'{model_name}' Performance\nOverall Score: {overall_score:.2f}%", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Context Length', fontsize=12)
    ax.set_ylabel('Document Depth (%)', fontsize=12)
    ax.set_xticklabels([f'{int(c)//1000}k' for c in pivot.columns], rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved heatmap to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Needle-in-a-Haystack results.")
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    
    df = pd.read_parquet(args.results_path)
    df["score"] = df.apply(lambda row: 100.0 if str(row["answer"]).strip().lower() in str(row["generated"]).strip().lower() else 0.0, axis=1)
    df["lang"] = df["title"].str.extract(r"_(en|kr)_\d+$")
    
    os.makedirs(args.output_dir, exist_ok=True)

    for lang in ["en", "kr"]:
        df_lang = df[df['lang'] == lang].groupby('title')['score'].mean().reset_index()
        if not df_lang.empty:
            model_name_lang = f"{args.model_name}_{lang.upper()}"
            df_lang = df_lang.rename(columns={"title": "dataset", "score": model_name_lang})
            save_path = os.path.join(args.output_dir, f"{model_name_lang}.png")
            visualize_heatmap(df_lang, save_path, model_name_lang)

if __name__ == "__main__":
    main()