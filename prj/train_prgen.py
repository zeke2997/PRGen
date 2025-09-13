import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
import pickle
from pathlib import Path
from tqdm import tqdm
import ruptures as rpt
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


class StructuralAnalysisResult:

    def __init__(self):
        self.anchors = {'head': None, 'tail': None}  
        self.aligned_length = 0
        self.cat_free_layout = []  
        self.entropy_profile = []  
        self.schedule_G = []  
        self.stripe_codebooks = {}  
        
class SACTVocab:
    def __init__(self):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.ESC = 3
        self.MISS = 4
        
        self.byte_start = 5
        self.byte_end = 260
        
        self.macro_start = 261
        self.token_to_id = {
            '[PAD]': self.PAD,
            '[BOS]': self.BOS,
            '[EOS]': self.EOS,
            '[ESC]': self.ESC,
            '[MISS]': self.MISS
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        for b in range(256):
            self.token_to_id[f'B{b:02X}'] = self.byte_start + b
            self.id_to_token[self.byte_start + b] = f'B{b:02X}'
        
        self.next_id = self.macro_start
        
    def add_stripe_token(self, stripe_id: int, fragment_id: int) -> int:
        token_name = f'<S{stripe_id}#{fragment_id}>'
        if token_name not in self.token_to_id:
            self.token_to_id[token_name] = self.next_id
            self.id_to_token[self.next_id] = token_name
            self.next_id += 1
        return self.token_to_id[token_name]
    
    def byte_to_id(self, byte_val: int) -> int:
        return self.byte_start + byte_val
    
    def __len__(self):
        return self.next_id


class StructuralAnalyzer:
    
    def __init__(self, kmer_sizes=[4, 5, 6], support_threshold=0.3, 
                 position_std_threshold=3.0, entropy_smooth_window=5):
        self.kmer_sizes = kmer_sizes
        self.support_threshold = support_threshold
        self.position_std_threshold = position_std_threshold
        self.entropy_smooth_window = entropy_smooth_window
    
    def discover_anchors(self, payloads: List[bytes], boundary_size: int = 20) -> Dict:
        anchors = {'head': None, 'tail': None}
        
        for boundary_type in ['head', 'tail']:
            best_kmer = None
            best_score = 0
            
            for k in self.kmer_sizes:
                kmer_positions = defaultdict(list)
                
                for idx, payload in enumerate(payloads):
                    if len(payload) < k:
                        continue
                    
                    if boundary_type == 'head':
                        search_region = payload[:min(boundary_size, len(payload))]
                    else:
                        search_region = payload[max(0, len(payload) - boundary_size):]
                    
                    for i in range(len(search_region) - k + 1):
                        kmer = search_region[i:i+k]
                        if boundary_type == 'head':
                            kmer_positions[kmer].append(i)
                        else:
                            kmer_positions[kmer].append(len(payload) - len(search_region) + i)
                
                for kmer, positions in kmer_positions.items():
                    support = len(positions) / len(payloads)
                    if support >= self.support_threshold:
                        std = np.std(positions)
                        if std <= self.position_std_threshold:
                            score = support / (std + 1)
                            if score > best_score:
                                best_score = score
                                best_kmer = (kmer, int(np.median(positions)))
            
            anchors[boundary_type] = best_kmer
        
        return anchors
    
    def monotone_resample(self, payload: bytes, target_length: int) -> bytes:
        if len(payload) == target_length:
            return payload
        
        source_indices = np.linspace(0, len(payload) - 1, target_length)
        resampled = bytearray()
        
        for idx in source_indices:
            resampled.append(payload[int(round(idx))])
        
        return bytes(resampled)
    
    def align_payloads(self, payloads: List[bytes], anchors: Dict) -> Tuple[np.ndarray, int]:
        prefix_lengths = []
        core_lengths = []
        suffix_lengths = []
        
        for payload in payloads:
            if anchors['head'] is not None and anchors['tail'] is not None:
                head_kmer, head_pos = anchors['head']
                tail_kmer, tail_pos = anchors['tail']
                
                head_idx = payload.find(head_kmer)
                tail_idx = payload.rfind(tail_kmer)
                
                if head_idx != -1 and tail_idx != -1:
                    prefix_lengths.append(head_idx)
                    core_lengths.append(tail_idx - head_idx)
                    suffix_lengths.append(len(payload) - tail_idx)
            else:
                third = len(payload) // 3
                prefix_lengths.append(third)
                core_lengths.append(third)
                suffix_lengths.append(len(payload) - 2 * third)
        
        L_prefix = int(np.median(prefix_lengths)) if prefix_lengths else 20
        L_core = int(np.median(core_lengths)) if core_lengths else 100
        L_suffix = int(np.median(suffix_lengths)) if suffix_lengths else 20
        L_star = L_prefix + L_core + L_suffix
        
        aligned_matrix = np.zeros((len(payloads), L_star), dtype=np.uint8)
        
        for i, payload in enumerate(payloads):
            if len(payload) == 0:
                continue
            
            if anchors['head'] is not None and anchors['tail'] is not None:
                head_kmer, _ = anchors['head']
                tail_kmer, _ = anchors['tail']
                head_idx = payload.find(head_kmer)
                tail_idx = payload.rfind(tail_kmer)
                
                if head_idx != -1 and tail_idx != -1:
                    prefix = payload[:head_idx]
                    core = payload[head_idx:tail_idx]
                    suffix = payload[tail_idx:]
                else:
                    third = len(payload) // 3
                    prefix = payload[:third]
                    core = payload[third:2*third]
                    suffix = payload[2*third:]
            else:
                third = len(payload) // 3
                prefix = payload[:third]
                core = payload[third:2*third]
                suffix = payload[2*third:]
            
            prefix_aligned = self.monotone_resample(prefix, L_prefix)
            core_aligned = self.monotone_resample(core, L_core)
            suffix_aligned = self.monotone_resample(suffix, L_suffix)
            
            aligned = prefix_aligned + core_aligned + suffix_aligned
            aligned_matrix[i, :len(aligned)] = list(aligned)[:L_star]
        
        return aligned_matrix, L_star
    
    def compute_entropy_profile(self, aligned_matrix: np.ndarray) -> np.ndarray:
        """Compute position-wise Shannon entropy"""
        n_positions = aligned_matrix.shape[1]
        entropy_profile = np.zeros(n_positions)
        
        for pos in range(n_positions):
            column = aligned_matrix[:, pos]
            counts = np.bincount(column, minlength=256)
            probs = counts / len(column)
            probs = probs[probs > 0]  
            entropy_profile[pos] = -np.sum(probs * np.log2(probs))
        
        kernel_size = self.entropy_smooth_window
        kernel = np.ones(kernel_size) / kernel_size
        entropy_profile = np.convolve(entropy_profile, kernel, mode='same')
        
        return entropy_profile
    
    def detect_stripes_pelt(self, entropy_profile: np.ndarray, pen: float = 3.0) -> List[Tuple]:
        """Use PELT changepoint detection to delineate stripes"""
        algo = rpt.Pelt(model="rbf", min_size=3, jump=1).fit(entropy_profile.reshape(-1, 1))
        changepoints = algo.predict(pen=pen)
        
        regions = []
        start = 0
        for cp in changepoints:
            if cp > start:
                mean_entropy = np.mean(entropy_profile[start:cp])
                regions.append((start, cp, mean_entropy))
            start = cp
        
        median_entropy = np.median(entropy_profile)
        
        stripe_id = 0
        layout = []
        for start, end, mean_ent in regions:
            if mean_ent < median_entropy:
                layout.append(('CAT', start, end, stripe_id))
                stripe_id += 1
            else:
                layout.append(('FREE', start, end))
        
        return layout
    
    def merge_adjacent_stripes(self, layout: List[Tuple], max_gap: int = 2) -> List[Tuple]:
        """Merge adjacent CAT stripes that are very close"""
        if not layout:
            return layout
        
        stripe_id = -1
        
        if layout[0][0] == 'CAT':
            stripe_id = 0
            merged = [('CAT', layout[0][1], layout[0][2], stripe_id)]
        else:
            merged = [layout[0]]
        
        for item in layout[1:]:
            prev = merged[-1]
            
            if prev[0] == 'CAT' and item[0] == 'CAT' and item[1] - prev[2] <= max_gap:
                merged[-1] = ('CAT', prev[1], item[2], prev[3])  
            else:
                if item[0] == 'CAT':
                    stripe_id += 1
                    merged.append(('CAT', item[1], item[2], stripe_id))
                else:
                    merged.append(item)
        
        return merged
    
    def build_stripe_codebooks(self, aligned_matrix: np.ndarray, layout: List[Tuple], 
                              coverage_threshold: float = 0.95, max_codebook_size: int = 1000) -> Dict:
        """Build stripe-specific codebooks via frequent pattern mining"""
        codebooks = {}
        
        for item in layout:
            if item[0] != 'CAT':
                continue
            
            _, start, end, stripe_id = item
            width = end - start
            
            fragments = []
            for row in aligned_matrix:
                fragment = bytes(row[start:end])
                fragments.append(fragment)
            
            fragment_counts = Counter(fragments)
            total = len(fragments)
            
            sorted_fragments = sorted(fragment_counts.items(), key=lambda x: x[1], reverse=True)
            
            codebook = {}
            covered = 0
            for frag, count in sorted_fragments:
                if len(codebook) >= max_codebook_size:
                    break
                codebook[frag] = len(codebook)
                covered += count
                if covered / total >= coverage_threshold:
                    break
            
            codebooks[stripe_id] = codebook
        
        return codebooks
    
    def analyze_group(self, payloads: List[bytes]) -> StructuralAnalysisResult:
        """Complete structural analysis for a group"""
        result = StructuralAnalysisResult()
        
        result.anchors = self.discover_anchors(payloads)
        
        aligned_matrix, L_star = self.align_payloads(payloads, result.anchors)
        result.aligned_length = L_star
        
        result.entropy_profile = self.compute_entropy_profile(aligned_matrix)
        
        raw_layout = self.detect_stripes_pelt(result.entropy_profile)
        result.cat_free_layout = self.merge_adjacent_stripes(raw_layout)
        
        result.stripe_codebooks = self.build_stripe_codebooks(
            aligned_matrix, result.cat_free_layout)
        
        result.schedule_G = []
        for item in result.cat_free_layout:
            if item[0] == 'CAT':
                _, start, end, stripe_id = item
                result.schedule_G.append(('STRIPE', stripe_id))
            else:
                _, start, end = item
                for _ in range(end - start):
                    result.schedule_G.append('BYTE')
        
        return result


class SACTTokenizer:
    
    def __init__(self, vocab: SACTVocab, structural_results: Dict[Tuple, StructuralAnalysisResult]):
        self.vocab = vocab
        self.structural_results = structural_results
        self.context_vocab = self._build_context_vocab()
    
    def _build_context_vocab(self) -> Dict[str, int]:
        context_vocab = {
            '[MISS]': self.vocab.MISS
        }
        next_id = 1000  
        
        all_values = set()
        for group_key, result in self.structural_results.items():
            protocol, port, path = group_key
            all_values.add(f'protocol:{protocol}')
            all_values.add(f'port:{port}')
            all_values.add(f'path:{path}')
        
        for val in sorted(all_values):
            if val not in context_vocab:
                context_vocab[val] = next_id
                next_id += 1
        
        return context_vocab
    
    def encode_context(self, protocol: str, port: int, path: str, 
                       brand: str, product: str, type_: str, firmware: str) -> List[int]:
        tokens = []
        
        tokens.append(self.context_vocab.get(f'protocol:{protocol}', self.vocab.MISS))
        tokens.append(self.context_vocab.get(f'port:{port}', self.vocab.MISS))
        tokens.append(self.context_vocab.get(f'path:{path}', self.vocab.MISS))
        
        for attr in [brand, product, type_, firmware]:
            if attr and attr.strip():
                attr_hash = hash(attr) % 10000
                tokens.append(attr_hash)
            else:
                tokens.append(self.vocab.MISS)
        
        return tokens
    
    def align_and_encode_payload(self, payload: bytes, group_key: Tuple) -> List[int]:
        result = self.structural_results[group_key]
        
        analyzer = StructuralAnalyzer()
        aligned_matrix, _ = analyzer.align_payloads([payload], result.anchors)
        aligned_payload = aligned_matrix[0]
        
        sact_tokens = [self.vocab.BOS]
        
        for item in result.cat_free_layout:
            if item[0] == 'CAT':
                _, start, end, stripe_id = item
                fragment = bytes(aligned_payload[start:end])
                
                codebook = result.stripe_codebooks[stripe_id]
                if fragment in codebook:
                    fragment_id = codebook[fragment]
                    macro_token_id = self.vocab.add_stripe_token(stripe_id, fragment_id)
                    sact_tokens.append(macro_token_id)
                else:
                    sact_tokens.append(self.vocab.ESC)
                    for byte_val in fragment:
                        sact_tokens.append(self.vocab.byte_to_id(byte_val))
            else:
                _, start, end = item
                for byte_val in aligned_payload[start:end]:
                    sact_tokens.append(self.vocab.byte_to_id(byte_val))
        
        sact_tokens.append(self.vocab.EOS)
        return sact_tokens


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class PRGenModel(nn.Module):
    
    def __init__(self, vocab_size: int, context_vocab_size: int, 
                 d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 2, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        self.context_embedding = nn.Embedding(context_vocab_size + 10000, d_model)
        self.context_pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.payload_embedding = nn.Embedding(vocab_size, d_model)
        self.payload_pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.payload_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, context_ids, payload_ids, tgt_mask=None, tgt_key_padding_mask=None):

        context_embedded = self.context_embedding(context_ids)
        context_embedded = self.context_pos_encoder(context_embedded)
        memory = self.context_encoder(context_embedded)
        
        payload_embedded = self.payload_embedding(payload_ids)
        payload_embedded = self.payload_pos_encoder(payload_embedded)
        
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(payload_ids.size(1)).to(payload_ids.device)
        
        decoder_output = self.payload_decoder(
            tgt=payload_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        logits = self.output_projection(decoder_output)
        return logits
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

class PRGenDataset(Dataset):
    
    def __init__(self, data_file: str, tokenizer: SACTTokenizer, 
                 structural_results: Dict, mode: str = 'train',
                 mask_rate: float = 0.0, entropy_exponent: float = 0.0):
        self.tokenizer = tokenizer
        self.structural_results = structural_results
        self.mode = mode
        self.mask_rate = mask_rate
        self.entropy_exponent = entropy_exponent
        
        self.samples = []
        with open(data_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                req_info = data['req_info']
                
                protocol = req_info['protocol']
                port = req_info['port']
                path = req_info['req_name']
                group_key = (protocol, port, path)
                
                if group_key not in structural_results:
                    continue
                
                payload_hex = data['res_payload']
                payload = bytes.fromhex(payload_hex)
                
                brand = req_info.get('brand', '')
                product = req_info.get('product', '')
                type_ = req_info.get('type', '')
                firmware = req_info.get('firmware', '')
                
                self.samples.append({
                    'group_key': group_key,
                    'payload': payload,
                    'context': (protocol, port, path, brand, product, type_, firmware)
                })
    
    def __len__(self):
        return len(self.samples)
    
    def apply_egm_masking(self, tokens: List[int], group_key: Tuple) -> Tuple[List[int], List[int]]:
        if self.mask_rate == 0:
            return tokens, tokens
        
        result = self.structural_results[group_key]
        entropy_profile = result.entropy_profile
        
        weights = []
        token_idx = 0
        for item in result.cat_free_layout:
            if item[0] == 'CAT':
                _, start, end, stripe_id = item
                mean_entropy = np.mean(entropy_profile[start:end])
                weights.append((1.0 + mean_entropy) ** self.entropy_exponent)
                token_idx += 1
            else:
                _, start, end = item
                for pos in range(start, end):
                    ent = entropy_profile[pos] if pos < len(entropy_profile) else 1.0
                    weights.append((1.0 + ent) ** self.entropy_exponent)
                    token_idx += 1
        
        weights = np.array(weights[:len(tokens)-2]) 
        weights = weights / (weights.sum() + 1e-8)
        
        n_mask = int(len(weights) * self.mask_rate)
        if n_mask > 0:
            mask_indices = np.random.choice(
                len(weights), size=n_mask, replace=False, p=weights
            )
            mask_indices = set(mask_indices)
        else:
            mask_indices = set()
        
        input_tokens = [tokens[0]]  
        target_tokens = [tokens[0]]
        
        for i, token in enumerate(tokens[1:-1]):
            if i in mask_indices:
                input_tokens.append(self.tokenizer.vocab.PAD)  
            else:
                input_tokens.append(token)
            target_tokens.append(token)
        
        input_tokens.append(tokens[-1])  
        target_tokens.append(tokens[-1])
        
        return input_tokens, target_tokens
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        context_ids = self.tokenizer.encode_context(*sample['context'])
        
        payload_tokens = self.tokenizer.align_and_encode_payload(
            sample['payload'], sample['group_key']
        )
        
        if self.mode == 'train' and self.mask_rate > 0:
            input_tokens, target_tokens = self.apply_egm_masking(
                payload_tokens, sample['group_key']
            )
            input_ids = input_tokens[:-1]  
            target_ids = target_tokens[1:]  
        else:
            input_ids = payload_tokens[:-1]  
            target_ids = payload_tokens[1:]   
        
        return {
            'context_ids': torch.tensor(context_ids, dtype=torch.long),
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }

def collate_fn(batch):
    """Collate function with padding"""
    max_context_len = max(len(b['context_ids']) for b in batch)
    max_seq_len = max(len(b['input_ids']) for b in batch)
    
    context_ids = torch.zeros(len(batch), max_context_len, dtype=torch.long)
    input_ids = torch.zeros(len(batch), max_seq_len, dtype=torch.long)
    target_ids = torch.zeros(len(batch), max_seq_len, dtype=torch.long)
    
    for i, b in enumerate(batch):
        context_ids[i, :len(b['context_ids'])] = b['context_ids']
        input_ids[i, :len(b['input_ids'])] = b['input_ids']
        target_ids[i, :len(b['target_ids'])] = b['target_ids']
    
    return {
        'context_ids': context_ids,
        'input_ids': input_ids,
        'target_ids': target_ids
    }


class PRGenTrainer:
    """Training pipeline for PRGen"""
    
    def __init__(self, model, tokenizer, structural_results, device, args):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.structural_results = structural_results
        self.device = device
        self.args = args
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = self._build_scheduler()
        
        self.scaler = GradScaler()
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab.PAD)
    
    def _build_scheduler(self):
        def lr_lambda(step):
            if step < self.args.warmup_steps:
                return step / self.args.warmup_steps
            else:
                progress = (step - self.args.warmup_steps) / (self.args.max_steps - self.args.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch):
        self.model.train()
        
        context_ids = batch['context_ids'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        
        with autocast():
            logits = self.model(context_ids, input_ids, tgt_mask=None)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        
        self.scaler.scale(loss).backward()
        
        return loss.item()
    
    def train_epoch(self, dataloader, epoch, stage='pretrain'):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'{stage.capitalize()} Epoch {epoch}')
        
        accum_steps = 0
        for batch in progress_bar:
            loss = self.train_step(batch)
            total_loss += loss
            accum_steps += 1
            
            if accum_steps >= self.args.gradient_accumulation_steps:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                accum_steps = 0
            
            progress_bar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'})
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc='Validating'):
            context_ids = batch['context_ids'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            with autocast():
                logits = self.model(context_ids, input_ids, tgt_mask=None)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, path, epoch, stage):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        torch.save({
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['stage']


def main():
    class Args:
        data_dir = 'dataset'
        train_file = 'dataset/train_dataset.json'  
        val_file = 'dataset/val_dataset.json'     
        output_dir = 'outputs'
        
        d_model = 512
        nhead = 8
        num_encoder_layers = 2
        num_decoder_layers = 6
        dim_feedforward = 2048
        dropout = 0.1
        
        pretrain_steps = 100000
        pretrain_mask_rate = 0.30
        pretrain_entropy_start = 0.0
        pretrain_entropy_end = 2.0
        
        finetune_steps = 15000
        finetune_lr_multiplier = 0.2
        
        learning_rate = 3e-4
        weight_decay = 0.01
        warmup_steps = 2000
        max_steps = 115000  
        gradient_accumulation_steps = 4
        batch_size = 64
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    args = Args()
    
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("PRGen Training Pipeline")
    print("=" * 80)
    
    print("\n[1/6] Loading training dataset...")
    print(f"Training file: {args.train_file}")
    
    if not Path(args.train_file).exists():
        print(f"\n⚠️  Warning: {args.train_file} not found!")
        print("Please run data_split.py first to create train/val/test splits:")
        print("  python data_split.py --input dataset/all_dataset.json --output-dir dataset")
        print("\nOr if you want to use all data (not recommended), create a symbolic link:")
        print(f"  ln -s all_dataset.json {args.train_file}")
        return
    
    grouped_data = defaultdict(list)
    with open(args.train_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            req_info = data['req_info']
            group_key = (req_info['protocol'], req_info['port'], req_info['req_name'])
            payload = bytes.fromhex(data['res_payload'])
            grouped_data[group_key].append(payload)
    
    print(f"Found {len(grouped_data)} unique groups in training data")
    total_train_samples = sum(len(v) for v in grouped_data.values())
    print(f"Total training samples: {total_train_samples:,}")
    
    print("\n[2/6] Running structural analysis on training groups...")
    analyzer = StructuralAnalyzer()
    structural_results = {}
    
    for group_key, payloads in tqdm(grouped_data.items(), desc="Analyzing groups"):
        if len(payloads) >= 100:  
            result = analyzer.analyze_group(payloads)
            structural_results[group_key] = result
    
    print(f"Analyzed {len(structural_results)} groups")
    
    with open(f"{args.output_dir}/structural_results.pkl", 'wb') as f:
        pickle.dump(structural_results, f)
    
    print("\n[3/6] Building vocabulary and tokenizer...")
    vocab = SACTVocab()
    
    for group_key, result in structural_results.items():
        for stripe_id, codebook in result.stripe_codebooks.items():
            for fragment_id in range(len(codebook)):
                vocab.add_stripe_token(stripe_id, fragment_id)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    tokenizer = SACTTokenizer(vocab, structural_results)
    
    with open(f"{args.output_dir}/tokenizer.pkl", 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("\n[4/6] Creating datasets...")
    
    pretrain_dataset = PRGenDataset(
        args.train_file, tokenizer, structural_results,
        mode='train', mask_rate=args.pretrain_mask_rate,
        entropy_exponent=args.pretrain_entropy_start
    )
    
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    finetune_dataset = PRGenDataset(
        args.train_file, tokenizer, structural_results,
        mode='train', mask_rate=0.0
    )
    
    finetune_loader = DataLoader(
        finetune_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    print(f"Training dataset size: {len(pretrain_dataset):,}")
    
    if Path(args.val_file).exists():
        print(f"Validation file found: {args.val_file}")
        val_dataset = PRGenDataset(
            args.val_file, tokenizer, structural_results,
            mode='val', mask_rate=0.0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        print(f"Validation dataset size: {len(val_dataset):,}")
    else:
        print(f"No validation file found at {args.val_file}")
        val_loader = None
    
    print("\n[5/6] Building model...")
    model = PRGenModel(
        vocab_size=len(vocab),
        context_vocab_size=len(tokenizer.context_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(args.device)
    
    print("\n[6/6] Training model...")
    trainer = PRGenTrainer(model, tokenizer, structural_results, args.device, args)
    
    print("\n" + "=" * 80)
    print("STAGE 1: Pretrain with Entropy-Guided Masking")
    print("=" * 80)
    
    num_pretrain_epochs = args.pretrain_steps // len(pretrain_loader) + 1
    step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_pretrain_epochs):
        if step >= args.pretrain_steps:
            break
        
        progress = step / args.pretrain_steps
        current_gamma = args.pretrain_entropy_start + progress * (
            args.pretrain_entropy_end - args.pretrain_entropy_start
        )
        pretrain_dataset.entropy_exponent = current_gamma
        
        avg_loss = trainer.train_epoch(pretrain_loader, epoch, stage='pretrain')
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, gamma={current_gamma:.2f}")
        
        if val_loader is not None and (epoch + 1) % 5 == 0:
            val_loss = trainer.validate(val_loader)
            print(f"  Validation loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint(
                    f"{args.output_dir}/checkpoint_pretrain_best.pt",
                    epoch, 'pretrain'
                )
                print(f"  ✓ New best validation loss!")
        
        step += len(pretrain_loader)
        
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(
                f"{args.output_dir}/checkpoint_pretrain_epoch{epoch}.pt",
                epoch, 'pretrain'
            )
    
    trainer.save_checkpoint(f"{args.output_dir}/checkpoint_pretrain_final.pt", 
                          num_pretrain_epochs, 'pretrain')
    
    print("\n" + "=" * 80)
    print("STAGE 2: Fine-tune without masking")
    print("=" * 80)
    
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = args.learning_rate * args.finetune_lr_multiplier
    
    num_finetune_epochs = args.finetune_steps // len(finetune_loader) + 1
    
    for epoch in range(num_finetune_epochs):
        avg_loss = trainer.train_epoch(finetune_loader, epoch, stage='finetune')
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}")
        
        if val_loader is not None and (epoch + 1) % 2 == 0:
            val_loss = trainer.validate(val_loader)
            print(f"  Validation loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint(
                    f"{args.output_dir}/checkpoint_finetune_best.pt",
                    epoch, 'finetune'
                )
                print(f"  ✓ New best validation loss!")
        
        if (epoch + 1) % 2 == 0:
            trainer.save_checkpoint(
                f"{args.output_dir}/checkpoint_finetune_epoch{epoch}.pt",
                epoch, 'finetune'
            )
    
    trainer.save_checkpoint(f"{args.output_dir}/checkpoint_final.pt", 
                          num_finetune_epochs, 'finetune')
    
    print("\n" + "=" * 80)
    print("Training completed!")
    if val_loader is not None:
        print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 80)

if __name__ == '__main__':
    main()
