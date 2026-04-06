"""
RecBole TedRec 실행 스크립트
- 베이스라인 조건에 맞춤
- Leave-two-out split
- CrossEntropy Loss
- 101 candidates evaluation
"""

import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_logger, get_trainer, set_color

from model import TedRec
from dataset import TedRecDataset


def run_tedrec(config_file='config.yaml', config_dict=None):
    """TedRec 학습 및 평가"""
    
    # ── Configuration ────────────────────────────────────────────
    config = Config(
        model=TedRec,
        config_file_list=[config_file],
        config_dict=config_dict
    )
    # Seed 설정 안 함 - 매번 다른 초기화
    
    # ── Logger ───────────────────────────────────────────────────
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    
    # ── Dataset ──────────────────────────────────────────────────
    dataset = TedRecDataset(config)
    logger.info(dataset)
    
    # ── Data Split ───────────────────────────────────────────────
    # RecBole의 leave-two-out split 사용
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # ── Model ────────────────────────────────────────────────────
    model = TedRec(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    # ── Trainer ──────────────────────────────────────────────────
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # ── Training ─────────────────────────────────────────────────
    best_valid_score, best_valid_result = trainer.fit(
        train_data, 
        valid_data, 
        saved=True, 
        show_progress=config['show_progress']
    )
    
    # ── Evaluation ───────────────────────────────────────────────
    test_result = trainer.evaluate(
        test_data, 
        load_best_model=True, 
        show_progress=config['show_progress']
    )
    
    # ── Results ──────────────────────────────────────────────────
    logger.info(set_color('Best Valid', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('Test Result', 'yellow') + f': {test_result}')
    
    # 결과 출력 (베이스라인 형식: HR, NDCG, MRR)
    print(f"\n{'='*60}")
    print(f"Best Valid NDCG@10: {best_valid_result['ndcg@10']:.4f}  "
          f"HR@10: {best_valid_result['hit@10']:.4f}  "
          f"MRR: {best_valid_result['mrr@10']:.4f}")
    print(f"Test Result NDCG@10: {test_result['ndcg@10']:.4f}  "
          f"HR@10: {test_result['hit@10']:.4f}  "
          f"MRR: {test_result['mrr@10']:.4f}")
    print(f"{'='*60}")
    
    return {
        'model': config['model'],
        'dataset': config['dataset'],
        'best_valid_score': best_valid_score,
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Config file path')
    parser.add_argument('--gpu', type=int, default=3,
                        help='GPU id to use')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Temperature for loss scaling')
    
    args = parser.parse_args()
    
    # Config dict override
    config_dict = {
        'gpu_id': args.gpu,
    }
    if args.temperature is not None:
        config_dict['temperature'] = args.temperature
    
    # Run
    results = run_tedrec(
        config_file=args.config,
        config_dict=config_dict
    )
    
    print("\n✅ Training completed!")