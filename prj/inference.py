import torch
import pickle
import json
from pathlib import Path
import argparse

from train_prgen import (
    PRGenModel, 
    SACTTokenizer, 
    SACTVocab, 
    StructuralAnalysisResult,
    StructuralAnalyzer
)


class PRGenGenerator:
    
    def __init__(self, model_path: str, tokenizer_path: str, device: str = 'cuda'):
        self.device = device
        
        print(f"Loading tokenizer from {tokenizer_path}...")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        vocab_size = len(self.tokenizer.vocab)
        context_vocab_size = len(self.tokenizer.context_vocab)
        
        self.model = PRGenModel(
            vocab_size=vocab_size,
            context_vocab_size=context_vocab_size,
            d_model=512,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    @torch.no_grad()
    def generate(self, protocol: str, port: int, path: str, 
                 brand: str = '', product: str = '', 
                 type_: str = '', firmware: str = '',
                 max_length: int = 512, temperature: float = 1.0,
                 top_k: int = 50, top_p: float = 0.9) -> bytes:

        group_key = (protocol, port, path)
        
        if group_key not in self.tokenizer.structural_results:
            print(f"Warning: Group {group_key} not found in training data.")
            print(f"Available groups: {list(self.tokenizer.structural_results.keys())[:5]}...")
            group_key = list(self.tokenizer.structural_results.keys())[0]
            print(f"Using fallback group: {group_key}")
        
        context_ids = self.tokenizer.encode_context(
            protocol, port, path, brand, product, type_, firmware
        )
        context_ids = torch.tensor([context_ids], dtype=torch.long).to(self.device)
        
        generated = [self.tokenizer.vocab.BOS]
        
        for _ in range(max_length):
            input_ids = torch.tensor([generated], dtype=torch.long).to(self.device)
            
            logits = self.model(context_ids, input_ids, tgt_mask=None)
            next_token_logits = logits[0, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == self.tokenizer.vocab.EOS:
                break
            
            generated.append(next_token)
        
        payload = self._decode_tokens(generated, group_key)
        return payload
    
    def _decode_tokens(self, tokens: list, group_key: tuple) -> bytes:
        payload = bytearray()
        result = self.tokenizer.structural_results.get(group_key)
        
        i = 1  
        while i < len(tokens):
            token = tokens[i]
            
            if token == self.tokenizer.vocab.EOS:
                break
            elif token == self.tokenizer.vocab.ESC:
                i += 1
                continue
            elif self.tokenizer.vocab.byte_start <= token <= self.tokenizer.vocab.byte_end:
                byte_val = token - self.tokenizer.vocab.byte_start
                payload.append(byte_val)
            elif token >= self.tokenizer.vocab.macro_start:
                token_name = self.tokenizer.vocab.id_to_token.get(token, '')
                if token_name.startswith('<S') and result is not None:
                    try:
                        parts = token_name[2:-1].split('#')
                        stripe_id = int(parts[0])
                        fragment_id = int(parts[1])
                        
                        if stripe_id in result.stripe_codebooks:
                            codebook = result.stripe_codebooks[stripe_id]
                            for fragment, frag_id in codebook.items():
                                if frag_id == fragment_id:
                                    payload.extend(fragment)
                                    break
                    except Exception as e:
                        pass
            
            i += 1
        
        return bytes(payload)


def main():
    parser = argparse.ArgumentParser(description='Generate probe-response payloads with PRGen')
    parser.add_argument('--model', type=str, default='outputs/checkpoint_final.pt')
    parser.add_argument('--tokenizer', type=str, default='outputs/tokenizer.pkl')
    parser.add_argument('--protocol', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--brand', type=str, default='')
    parser.add_argument('--product', type=str, default='')
    parser.add_argument('--type', type=str, default='')
    parser.add_argument('--firmware', type=str, default='')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--output', type=str, default='generated_payloads.json')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    generator = PRGenGenerator(args.model, args.tokenizer, args.device)
    
    print(f"\nGenerating {args.num_samples} payloads...")
    print(f"Protocol: {args.protocol}, Port: {args.port}, Path: {args.path}")
    print(f"Temperature: {args.temperature}")
    print("-" * 80)
    
    results = []
    for i in range(args.num_samples):
        payload = generator.generate(
            protocol=args.protocol,
            port=args.port,
            path=args.path,
            brand=args.brand,
            product=args.product,
            type_=args.type,
            firmware=args.firmware,
            temperature=args.temperature
        )
        
        payload_hex = payload.hex()
        print(f"\nSample {i+1}:")
        print(f"  Length: {len(payload)} bytes")
        print(f"  Hex: {payload_hex[:100]}{'...' if len(payload_hex) > 100 else ''}")
        
        results.append({
            'sample_id': i + 1,
            'protocol': args.protocol,
            'port': args.port,
            'path': args.path,
            'brand': args.brand,
            'product': args.product,
            'type': args.type,
            'firmware': args.firmware,
            'payload_hex': payload_hex,
            'payload_length': len(payload)
        })
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"Generated {args.num_samples} payloads")
    print(f"Results saved to: {args.output}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
