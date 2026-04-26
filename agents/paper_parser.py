"""
Paper parsing agent - extracts structured information from PDFs.
"""

import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from reproagent.models import LLMClient
from reproagent.state import PaperState


class PaperParser:
    """
    Parses research papers and extracts key information.
    Uses LLM to extract structured data from paper text.
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Args:
            llm_client: LLM client for extraction
        """
        self.llm = llm_client
    
    def parse_paper(self, pdf_path: str) -> PaperState:
        """
        Parse paper and extract structured information.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PaperState with extracted info
        """
        print(f"📄 Parsing paper: {pdf_path}")
        
        # Extract text from PDF
        text = self._extract_text(pdf_path)
        
        if not text or text.startswith("Error"):
            print(f"❌ Failed to extract text from PDF")
            return PaperState(pdf_path=pdf_path)
        
        print(f"✅ Extracted {len(text)} characters")
        
        # Extract structured info with LLM
        extracted = self._extract_with_llm(text)
        
        # Build PaperState
        state = PaperState(
            pdf_path=pdf_path,
            title=extracted.get('title', ''),
            abstract=extracted.get('abstract', ''),
            dataset=extracted.get('dataset', ''),
            model=extracted.get('model', ''),
            target_metric=float(extracted.get('target_metric', 0.0)),
            metric_name=extracted.get('metric_name', 'accuracy'),
            github_links=extracted.get('github_links', []),
            key_claims=extracted.get('key_claims', []),
            parsed=True,
            confidence=extracted.get('confidence', 0.8)
        )
        
        print(f"✅ Paper parsed: {state.title}")
        print(f"   Dataset: {state.dataset}")
        print(f"   Model: {state.model}")
        print(f"   Target: {state.target_metric} {state.metric_name}")
        
        return state
    
    def _extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF.
        Tries multiple methods.
        """
        try:
            # Try PyPDF2 first (faster)
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                # Extract first 10 pages
                for page in reader.pages[:10]:
                    text += page.extract_text() + "\n"
                return text
                
        except Exception as e:
            print(f"⚠️  PyPDF2 failed: {e}")
            
            try:
                # Try pdfplumber (more accurate)
                import pdfplumber
                
                text = ""
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages[:10]:
                        text += page.extract_text() + "\n"
                return text
                
            except Exception as e2:
                print(f"⚠️  pdfplumber failed: {e2}")
                return f"Error: Could not extract text from PDF"
    
    def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Use LLM to extract structured information.
        
        Args:
            text: Paper text
            
        Returns:
            Extracted information dict
        """
        # Truncate text to fit in context
        text_sample = text[:5000]
        
        prompt = f"""
Extract the following information from this research paper:

1. **Title**: Full paper title
2. **Abstract**: Paper abstract (if present)
3. **Dataset**: Dataset used (e.g., "CIFAR-10", "ImageNet", "COCO")
4. **Model**: Model architecture (e.g., "ResNet-50", "BERT", "GPT-2")
5. **Target Metric**: Best reported performance value as a number. Extract exactly what is in the text.
6. **Metric Name**: Type of metric (e.g., "FID", "accuracy", "CLIP score", "BLEU"). DO NOT default to accuracy!
7. **GitHub Links**: Any GitHub URLs mentioned (full URLs)
8. **Key Claims**: Main performance claims (list)

Paper excerpt:
{text_sample}

Respond with ONLY valid JSON in this exact format:
{{
    "title": "paper title here",
    "abstract": "abstract text here",
    "dataset": "dataset name",
    "model": "model name",
    "target_metric": 12.34,
    "metric_name": "FID",
    "github_links": ["https://github.com/user/repo"],
    "key_claims": ["claim 1", "claim 2"],
    "confidence": 0.9
}}
"""
        
        try:
            result = self.llm.generate_structured(prompt)
            
            # Validate and clean result
            if 'error' not in result:
                # Ensure github_links is a list
                if 'github_links' in result and isinstance(result['github_links'], str):
                    result['github_links'] = [result['github_links']]
                
                # Extract GitHub links from text if none found
                if not result.get('github_links'):
                    result['github_links'] = self._extract_github_links(text)
                
                return result
            else:
                print(f"⚠️  LLM extraction failed: {result.get('error')}")
                
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
        
        # Fallback: regex extraction
        return self._fallback_extraction(text)
    
    def _extract_github_links(self, text: str) -> List[str]:
        """Extract GitHub URLs using regex."""
        pattern = r'https?://github\.com/[\w\-]+/[\w\-]+'
        matches = re.findall(pattern, text)
        return list(set(matches))  # unique links
    
    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """
        Fallback extraction using simple heuristics.
        Used when LLM fails.
        """
        print("⚠️  Using fallback extraction")
        
        # Extract title (usually first line or after "Title:")
        title = ""
        lines = text.split('\n')
        for line in lines[:20]:
            if line.strip() and len(line.strip()) > 10:
                title = line.strip()
                break
        
        # Extract dataset mentions
        dataset = ""
        dataset_patterns = [
            r'(CIFAR-10|CIFAR-100|ImageNet|COCO|MNIST|Fashion-MNIST)',
            r'(?:on|using|dataset)\s+(\w+)',
        ]
        for pattern in dataset_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                dataset = match.group(1)
                break
        
        # Extract model mentions
        model = ""
        model_patterns = [
            r'(ResNet-\d+|VGG-\d+|BERT|GPT-\d+|Transformer)',
            r'(AlexNet|DenseNet|MobileNet|EfficientNet)',
        ]
        for pattern in model_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                model = match.group(1)
                break
        
        # Extract metrics
        metric_pattern = r'(\d+\.?\d*)\s*%?\s*(accuracy|precision|recall|F1|BLEU)'
        metric_match = re.search(metric_pattern, text, re.IGNORECASE)
        
        target_metric = 0.0
        metric_name = "accuracy"
        
        if metric_match:
            target_metric = float(metric_match.group(1))
            metric_name = metric_match.group(2).lower()
            
            # Convert percentage to decimal
            if target_metric > 1.0:
                target_metric = target_metric / 100.0
        
        # GitHub links
        github_links = self._extract_github_links(text)
        
        return {
            'title': title or "Unknown Paper",
            'abstract': "",
            'dataset': dataset or "Unknown",
            'model': model or "Unknown",
            'target_metric': target_metric,
            'metric_name': metric_name,
            'github_links': github_links,
            'key_claims': [],
            'confidence': 0.5
        }
    
    def parse_from_arxiv(self, arxiv_id: str) -> PaperState:
        """
        Parse paper from ArXiv ID.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., "2103.00020")
            
        Returns:
            PaperState
        """
        print(f"📄 Fetching paper from ArXiv: {arxiv_id}")
        
        try:
            import requests
            
            # Fetch ArXiv metadata
            url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            response = requests.get(url)
            
            if response.status_code == 200:
                # Parse XML response
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                
                # Extract metadata
                entry = root.find('{http://www.w3.org/2005/Atom}entry')
                
                if entry:
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                    abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                    
                    # Use LLM to extract technical details from abstract
                    extracted = self._extract_with_llm(f"Title: {title}\n\nAbstract: {abstract}")
                    
                    return PaperState(
                        pdf_path=f"arxiv:{arxiv_id}",
                        title=title,
                        abstract=abstract,
                        dataset=extracted.get('dataset', ''),
                        model=extracted.get('model', ''),
                        target_metric=extracted.get('target_metric', 0.0),
                        metric_name=extracted.get('metric_name', 'accuracy'),
                        github_links=extracted.get('github_links', []),
                        key_claims=extracted.get('key_claims', []),
                        parsed=True,
                        confidence=0.7
                    )
            
        except Exception as e:
            print(f"❌ ArXiv fetch failed: {e}")
        
        return PaperState(pdf_path=f"arxiv:{arxiv_id}")


# Test
if __name__ == "__main__":
    from reproagent.models import LLMClient
    
    llm = LLMClient()
    parser = PaperParser(llm)
    
    # Test with sample text
    sample_text = """
    Deep Residual Learning for Image Recognition
    
    Abstract: We present a residual learning framework to ease the training of networks 
    that are substantially deeper than those used previously. We achieve 95.2% accuracy 
    on CIFAR-10 dataset using ResNet-50 architecture.
    
    Code: https://github.com/example/resnet-cifar10
    """
    
    result = parser._extract_with_llm(sample_text)
    print(result)
