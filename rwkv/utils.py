########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

class PIPELINE_ARGS():
    def __init__(self, temperature=1.0, top_p=0.85, top_k=0, alpha_frequency=0.2, alpha_presence=0.2, alpha_decay=0.996, token_ban=[], token_stop=[], chunk_len=256):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alpha_frequency = alpha_frequency # Frequency Penalty (as in GPT-3)
        self.alpha_presence = alpha_presence # Presence Penalty (as in GPT-3)
        self.alpha_decay = alpha_decay # gradually decay the penalty
        self.token_ban = token_ban # ban the generation of some tokens
        self.token_stop = token_stop # stop generation whenever you see any token here
        self.chunk_len = chunk_len # split input into chunks to save VRAM (shorter -> slower)

class PIPELINE():
    def __init__(self, model, WORD_NAME):
        self.model = model
        if WORD_NAME == 'cl100k_base':
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(WORD_NAME)
        elif WORD_NAME == 'rwkv_vocab_v20230424':
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from rwkv_tokenizer import TRIE_TOKENIZER
            self.tokenizer = TRIE_TOKENIZER(os.path.dirname(os.path.abspath(__file__)) + '/rwkv_vocab_v20230424.txt')        
        else:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(WORD_NAME)

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def encode(self, x):
        if 'Tokenizer' in str(type(self.tokenizer)):
            return self.tokenizer.encode(x).ids
        else:
            return self.tokenizer.encode(x)
    
    def decode(self, x):
        return self.tokenizer.decode(x)

    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        if temperature == 0:
            temperature = 1.0
            top_p = 0
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        # 'privateuseone' is the type of custom devices like `torch_directml.device()`
        if probs.device.type in ['cpu', 'privateuseone']:
            probs = probs.cpu().numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)
    
    def generate(self, ctx, token_count=100, args=PIPELINE_ARGS(), callback=None, state=None):
        all_tokens = []
        out_last = 0
        out_str = ''
        occurrence = {}
        for i in range(token_count):

            # forward & adjust prob.
            tokens = self.encode(ctx) if i == 0 else [token]
            while len(tokens) > 0:
                out, state = self.model.forward(tokens[:args.chunk_len], state)
                tokens = tokens[args.chunk_len:]
                
            for n in args.token_ban:
                out[n] = -float('inf')
            for n in occurrence:
                out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)
            
            # sampler
            token = self.sample_logits(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            if token in args.token_stop:
                break
            all_tokens += [token]
            for xxx in occurrence:
                occurrence[xxx] *= args.alpha_decay
            
            ttt = self.decode([token])
            www = 1
            if ttt in ' \t0123456789':
                www = 0
            # elif ttt in '\r\n,.;?!"\':+-*/=#@$%^&_`~|<>\\()[]{}，。；“”：？！（）【】':
            #     www = 0.5
            if token not in occurrence:
                occurrence[token] = www
            else:
                occurrence[token] += www
            # print(occurrence) # debug
            
            # output
            tmp = self.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # is valid utf-8 string?
                if callback:
                    callback(tmp)
                out_str += tmp
                out_last = i + 1
        return out_str
    def encode_bsz(self, x):

        list = []
        max_len = 0
        for i in x:
            t= self.tokenizer.encode(i)
            max_len = max(len(t),max_len)
            #t = torch.tensor(t)
            list.append(t)
        padded_sequences = []
        masks = []
        for seq in list:
            padding_length = max_len - len(seq)
            padded_sequences.append([0] * padding_length + seq)
            masks.append([0] * padding_length + [1] * len(seq))

        padded_sequences = torch.tensor(padded_sequences)
        masks = torch.tensor(masks)

        return padded_sequences, masks
    # def decode_bsz(self, x):
    #     list = []
    #     for i in x:
    #         i = [int(i)]
    #         t = self.tokenizer.decode(i)
    #         list.append(t)
    #     return np.array(list, dtype='U')

    def decode_bsz(self, x, callback=None):    
        # 初始化batch的token缓存和解码位置
        if not hasattr(self, '_token_buffers'):
            self._token_buffers = [[] for _ in range(len(x))]
            self._out_last = [0 for _ in range(len(x))]
            self._out_str = ['' for _ in range(len(x))]  # 为每个batch维护完整输出
        
        # 如果batch大小发生变化，重新初始化缓存
        if len(self._token_buffers) != len(x):
            self._token_buffers = [[] for _ in range(len(x))]
            self._out_last = [0 for _ in range(len(x))]
            self._out_str = ['' for _ in range(len(x))]
        
        decoded_list = []
        for idx, token in enumerate(x):
            # 添加新token到对应batch的缓存
            self._token_buffers[idx].append(int(token))
            
            # 尝试解码未解码的tokens
            try:
                tmp = self.decode(self._token_buffers[idx][self._out_last[idx]:])
                
                # 检查是否是有效的UTF-8字符串
                if '\ufffd' not in tmp:
                    if callback:
                        # 为每个batch单独调用callback
                        callback(tmp, batch_idx=idx)
                    
                    # 更新该batch的完整输出和解码位置
                    self._out_str[idx] += tmp
                    self._out_last[idx] = len(self._token_buffers[idx])
                    decoded_list.append(tmp)
                else:
                    # 如果解码无效，返回空字符串
                    decoded_list.append('')
            except Exception as e:
                print(f"Error decoding batch {idx}: {e}")
                decoded_list.append('')
                continue
        
        return np.array(decoded_list, dtype='U')

    def reset_cache(self):
        """重置所有batch的缓存"""
        if hasattr(self, '_token_buffers'):
            del self._token_buffers
            del self._out_last
            del self._out_str

    def get_batch_output(self, batch_idx):
        """获取指定batch的完整输出"""
        if hasattr(self, '_out_str'):
            return self._out_str[batch_idx]
        return ''

    def sample_bsz(self, logits, temperature=1.0, top_p=0.5, top_k=0):
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = sorted_probs[torch.arange(probs.shape[0]),np.argmax(cumulative_probs > top_p,1)]
        probs[probs < cutoff.unsqueeze(1)] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[top_k:]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[:, 0]
        out = out.unsqueeze(1).cpu()
        return out
    
    def gen_bsz(self, ctx, token_count=100, args=PIPELINE_ARGS(), callback=None, state=None):
        B = len(ctx)
        all_str = {}
        all_state = {}
        set_n = np.arange(B)
        out_np = np.empty((B,), dtype='U')
        
        # 为每个batch创建occurrence字典
        occurrences = [{} for _ in range(B)]
        all_tokens = [[] for _ in range(B)]
        
        for i in range(token_count):
            # forward & adjust prob.
            if i == 0:
                tokens, mask = self.encode_bsz(ctx)
            else:
                tokens, mask = token, None
            
            out, state = self.model.forward(tokens, state, mask=mask)
            
            # 对每个batch应用penalty
            for b in range(len(set_n)):
                # 应用token_ban
                for n in args.token_ban:
                    out[b, n] = -float('inf')
                
                # 应用frequency和presence penalty
                for n in occurrences[b]:
                    out[b, n] -= (args.alpha_presence + occurrences[b][n] * args.alpha_frequency)
            
            # 采样
            token = self.sample_bsz(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            tmp = self.decode_bsz(token, callback=callback)
            
            # 更新occurrence和tokens
            for b in range(len(set_n)):
                if token[b].item() in args.token_stop:
                    # 处理停止token
                    all_str[set_n[b]] = out_np[b]
                    state_list = [s[b] for s in state]
                    all_state[set_n[b]] = state_list
                    continue
                    
                all_tokens[b].append(token[b].item())
                
                # 更新occurrence
                ttt = self.decode([token[b].item()])
                www = 1
                if ttt in ' \t0123456789':
                    www = 0
                    
                tok = token[b].item()
                if tok not in occurrences[b]:
                    occurrences[b][tok] = www
                else:
                    occurrences[b][tok] += www
                
                # 衰减所有penalty
                for xxx in occurrences[b]:
                    occurrences[b][xxx] *= args.alpha_decay
            
            k = len(tmp)-1
            while k >= 0:
                if '🐕' in tmp[k] or '\ufffd' in tmp[k] or '\n\n' in out_np[k]:
                    all_str[set_n[k]] = out_np[k]
                    state_list = []
                    for t, s in enumerate(state):
                        state_list.append(s[k])
                        if k == len(tmp) - 1:
                            state[t] = state[t][:-1, :]
                        else:
                            state[t] = torch.cat((state[t][:k, :], state[t][k + 1:, :]), dim=0)
                    all_state[set_n[k]] = state_list
                    set_n = np.delete(set_n, k, axis=0)
                    out_np = np.delete(out_np, k, axis=0)
                    token = np.delete(token, k, axis=0)
                    tmp = np.delete(tmp, k, axis=0)
                    # 删除对应的occurrence和tokens
                    del occurrences[k]
                    del all_tokens[k]
                    if len(set_n) == 0:
                        return all_str, all_state
                k -= 1
            out_np = np.char.add(out_np, tmp)
            
            # 如果是最后一个token，保存所有剩余序列
            if i == token_count - 1:
                for k in range(len(set_n)):
                    all_str[set_n[k]] = out_np[k]
                    state_list = []
                    for t, s in enumerate(state):
                        state_list.append(s[k])
                    all_state[set_n[k]] = state_list
                break
        
        return all_str, all_state
