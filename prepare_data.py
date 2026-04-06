"""
RecBole 데이터 준비 스크립트
- interaction.parquet → .inter 파일
- title_emb.npy → .text_feat.npy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def prepare_recbole_data(
    interaction_path='./interaction.parquet',
    title_npy_path='./title_emb.npy',
    output_dir='./dataset'
):
    """RecBole 형식으로 데이터 변환"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    dataset_name = 'microvideo'
    
    # ── 1. Interaction 데이터 로드 ────────────────────────────────
    print("Loading interaction data...")
    df = pd.read_parquet(interaction_path)
    df = df.sort_values(by=['user_id', 'timestamp'])
    
    # ── 2. .inter 파일 생성 ───────────────────────────────────────
    print("Creating .inter file...")
    inter_df = df[['user_id', 'item_id', 'timestamp']].copy()
    inter_df.columns = ['user_id:token', 'item_id:token', 'timestamp:float']
    
    inter_path = output_dir / f'{dataset_name}.inter'
    inter_df.to_csv(inter_path, sep='\t', index=False)
    print(f"Saved: {inter_path}")
    
    # ── 3. Text features (title_emb.npy) ──────────────────────────
    print("Preparing text features...")
    title_emb = np.load(title_npy_path)  # (num_items, 768)
    
    # RecBole 형식으로 저장
    text_feat_path = output_dir / f'{dataset_name}.text_feat.npy'
    np.save(text_feat_path, title_emb)
    print(f"Saved: {text_feat_path}")
    print(f"Text feat shape: {title_emb.shape}")
    
    # ── 4. 통계 정보 ──────────────────────────────────────────────
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].max() + 1
    num_interactions = len(df)
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Interactions: {num_interactions}")
    print(f"Sparsity: {1 - num_interactions / (num_users * num_items):.4f}")
    
    return dataset_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interaction', type=str, default='./interaction.parquet')
    parser.add_argument('--title', type=str, default='./title_emb.npy')
    parser.add_argument('--output', type=str, default='./dataset')
    args = parser.parse_args()
    
    dataset_name = prepare_recbole_data(
        interaction_path=args.interaction,
        title_npy_path=args.title,
        output_dir=args.output
    )
    print(f"\n✅ Dataset '{dataset_name}' prepared for RecBole!")