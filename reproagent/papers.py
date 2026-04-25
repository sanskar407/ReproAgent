"""
Paper dataset and loading utilities.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class PaperDataset:
    """
    Manages collection of research papers for reproduction.
    Organizes by difficulty level.
    """
    
    def __init__(self, data_dir: str = "data/papers"):
        self.data_dir = Path(data_dir)
        self.papers = self._load_papers()
    
    def _load_papers(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load papers from data directory."""
        papers = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        for difficulty in ['easy', 'medium', 'hard']:
            difficulty_dir = self.data_dir / difficulty
            
            if difficulty_dir.exists():
                for paper_file in difficulty_dir.glob('*.json'):
                    try:
                        with open(paper_file) as f:
                            paper_data = json.load(f)
                            paper_data['difficulty'] = difficulty
                            papers[difficulty].append(paper_data)
                    except Exception as e:
                        print(f"⚠️  Failed to load {paper_file}: {e}")
        
        return papers
    
    def get_paper(
        self,
        difficulty: Optional[str] = None,
        index: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Get a specific paper."""
        if difficulty:
            papers_list = self.papers.get(difficulty, [])
            if index < len(papers_list):
                return papers_list[index]
        else:
            # Get any paper
            all_papers = []
            for papers_list in self.papers.values():
                all_papers.extend(papers_list)
            
            if index < len(all_papers):
                return all_papers[index]
        
        return None
    
    def get_random_paper(self, difficulty: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get random paper."""
        import random
        
        if difficulty:
            papers_list = self.papers.get(difficulty, [])
        else:
            papers_list = []
            for plist in self.papers.values():
                papers_list.extend(plist)
        
        if papers_list:
            return random.choice(papers_list)
        
        return None
    
    def count(self, difficulty: Optional[str] = None) -> int:
        """Count papers."""
        if difficulty:
            return len(self.papers.get(difficulty, []))
        else:
            return sum(len(plist) for plist in self.papers.values())


# Example paper template
SAMPLE_PAPER_TEMPLATE = {
    "title": "ResNet-50 on CIFAR-10",
    "dataset": "CIFAR-10",
    "model": "ResNet-50",
    "target_metric": 0.95,
    "metric_name": "accuracy",
    "github_url": "https://github.com/example/resnet-cifar10",
    "key_claims": [
        "Achieves 95% accuracy on CIFAR-10",
        "Uses standard data augmentation",
        "Trains in 200 epochs"
    ],
    "ground_truth_config": {
        "learning_rate": 0.0001,
        "batch_size": 64,
        "optimizer": "adamw",
        "epochs": 200,
        "weight_decay": 0.01,
        "scheduler": "cosine"
    }
}


def create_sample_papers():
    """Create sample paper dataset for all difficulty levels."""
    data_dir = Path("data/papers")
    
    # --- EASY papers ---
    easy_dir = data_dir / "easy"
    easy_dir.mkdir(parents=True, exist_ok=True)
    
    easy_paper_1 = SAMPLE_PAPER_TEMPLATE.copy()
    easy_paper_1["difficulty"] = "easy"
    with open(easy_dir / "resnet_cifar10.json", 'w') as f:
        json.dump(easy_paper_1, f, indent=2)
    
    easy_paper_2 = {
        "title": "Simple CNN for MNIST Digit Classification",
        "dataset": "MNIST",
        "model": "CNN-Small",
        "target_metric": 0.99,
        "metric_name": "accuracy",
        "github_url": "https://github.com/pytorch/examples",
        "key_claims": [
            "Achieves 99% accuracy on MNIST",
            "Simple 2-layer CNN architecture",
            "Trains in under 10 epochs"
        ],
        "ground_truth_config": {
            "learning_rate": 0.001,
            "batch_size": 64,
            "optimizer": "adam",
            "epochs": 10
        },
        "difficulty": "easy"
    }
    with open(easy_dir / "mnist_cnn.json", 'w') as f:
        json.dump(easy_paper_2, f, indent=2)
    
    # --- MEDIUM papers ---
    medium_dir = data_dir / "medium"
    medium_dir.mkdir(parents=True, exist_ok=True)
    
    medium_paper = {
        "title": "Fine-tuning BERT for Text Classification",
        "dataset": "GLUE-SST2",
        "model": "BERT-base",
        "target_metric": 0.92,
        "metric_name": "accuracy",
        "github_url": "https://github.com/huggingface/transformers",
        "key_claims": [
            "Achieves 92% accuracy on SST-2",
            "Fine-tunes pre-trained BERT-base",
            "Requires careful learning rate tuning"
        ],
        "ground_truth_config": {
            "learning_rate": 2e-5,
            "batch_size": 32,
            "optimizer": "adamw",
            "epochs": 3,
            "warmup_steps": 500,
            "weight_decay": 0.01
        },
        "difficulty": "medium"
    }
    with open(medium_dir / "bert_finetuning.json", 'w') as f:
        json.dump(medium_paper, f, indent=2)
    
    # --- HARD papers ---
    hard_dir = data_dir / "hard"
    hard_dir.mkdir(parents=True, exist_ok=True)
    
    hard_paper = {
        "title": "Progressive GAN for High-Resolution Image Generation",
        "dataset": "CelebA-HQ",
        "model": "ProGAN",
        "target_metric": 0.85,
        "metric_name": "FID_score_inverse",
        "github_url": "",
        "key_claims": [
            "Generates 1024x1024 face images",
            "Uses progressive training curriculum",
            "Achieves FID of 7.3 on CelebA-HQ"
        ],
        "ground_truth_config": {
            "learning_rate": 0.001,
            "batch_size": 16,
            "optimizer": "adam",
            "epochs": 600,
            "latent_dim": 512,
            "beta1": 0.0,
            "beta2": 0.99
        },
        "difficulty": "hard"
    }
    with open(hard_dir / "gan_generation.json", 'w') as f:
        json.dump(hard_paper, f, indent=2)
    
    print("[OK] Sample papers created (easy: 2, medium: 1, hard: 1)")


if __name__ == "__main__":
    create_sample_papers()
