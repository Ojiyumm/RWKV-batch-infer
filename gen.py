import os, sys, torch, time
import numpy as np
from typing import List
import argparse

from colorama import init, Fore, Style


init()
def print_stream(text, batch_idx):
    """流式输出，使用list管理每个batch的输出"""
    colors = [
        Fore.LIGHTGREEN_EX,  # 亮绿色
        Fore.LIGHTBLUE_EX,   # 亮蓝色
        Fore.LIGHTYELLOW_EX, # 亮黄色
        Fore.LIGHTMAGENTA_EX,# 亮紫色
        Fore.LIGHTCYAN_EX,   # 亮青色
        Fore.LIGHTRED_EX,    # 亮红色
        Fore.LIGHTWHITE_EX,  # 亮白色
        Fore.BLUE,           # 蓝色
    ]
    
    # 初始化状态和batch输出列表
    if not hasattr(print_stream, 'started'):
        print_stream.started = set()
        print_stream.line_positions = {}  # 记录每个batch的行位置
        print_stream.total_lines = 0
        print_stream.batch_outputs = {}   # 存储每个batch的输出列表
    
    # 如果是新的batch，初始化其输出列表和位置
    if batch_idx not in print_stream.started:
        print_stream.line_positions[batch_idx] = print_stream.total_lines
        print_stream.batch_outputs[batch_idx] = []
        print(f"{colors[batch_idx % len(colors)]}[Batch {batch_idx}]: {Style.RESET_ALL}")
        print_stream.started.add(batch_idx)
        print_stream.total_lines += 1
    
    # 将新token添加到对应batch的输出列表
    print_stream.batch_outputs[batch_idx].append(text)
    
    # 保存当前光标位置
    print('\033[s', end='')
    
    # 移动到对应batch的行
    lines_up = print_stream.total_lines - print_stream.line_positions[batch_idx] - 1
    if lines_up > 0:
        print(f'\033[{lines_up}A', end='')
    
    # 清除当前行
    print('\033[2K', end='')
    print('\033[0G', end='')
    
    # 打印该batch的完整输出
    batch_text = ''.join(print_stream.batch_outputs[batch_idx])
    print(f"{colors[batch_idx % len(colors)]}[Batch {batch_idx}]: {batch_text}{Style.RESET_ALL}", end='', flush=True)
    
    # 恢复光标位置
    print('\033[u', end='')

def reset_print_stream():
    """重置print_stream的状态"""
    if hasattr(print_stream, 'started'):
        delattr(print_stream, 'started')
        delattr(print_stream, 'line_positions')
        delattr(print_stream, 'total_lines')

    
class StreamWriter:

    """多文件流式输出"""
    def __init__(self, num_batches, output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        self.files = [
            open(f"{output_dir}/output_{i}.txt", "w", encoding="utf-8") 
            for i in range(num_batches)
        ]
    
    def callback(self, text, batch_idx):
        self.files[batch_idx].write(text)
        self.files[batch_idx].flush()
    
    def close(self):
        for f in self.files:
            f.close()
def batch_inference(ctx_list: List[str], args, callback=None):
    """
    批量推理函数
    
    Args:
        ctx_list: 输入上下文列表
        args: 解析后的命令行参数
    
    Returns:
        tuple: (answer, state) 生成的答案和状态
    """
    # 设置环境变量
    os.environ['RWKV_JIT_ON'] = args.jit_on
    os.environ["RWKV_CUDA_ON"] = args.cuda_on
    os.environ["RWKV_fla_ON"] = args.fla_on
    
    # 导入必要的模块
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE, PIPELINE_ARGS
    
    # 初始化模型和pipeline
    model = RWKV(model=args.base_model, strategy='cuda fp16')
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    
    # 加载状态
    states = torch.load(args.state_file)
    states_value = []
    
    # 初始化状态
    for i in range(model.args.n_layer):
        key = f'blocks.{i}.att.time_state'
        value = states[key]
        
        prev_x = torch.zeros(args.batch_size, model.args.n_embd, 
                           device=args.device, dtype=torch.float16)
        prev_states = value.clone().detach().to(
            device=args.device, 
            dtype=torch.float32
        ).transpose(1, 2).expand(args.batch_size, *value.shape)
        prev_ffn = torch.zeros(args.batch_size, model.args.n_embd, 
                             device=args.device, dtype=torch.float16)
        
        states_value.append(prev_x)
        states_value.append(prev_states)
        states_value.append(prev_ffn)
    
    # 设置生成参数
    pipeline_args = PIPELINE_ARGS(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        alpha_frequency=args.alpha_frequency,
        alpha_presence=args.alpha_presence,
        alpha_decay=args.alpha_decay,
        token_ban=args.token_ban,
        token_stop=args.token_stop,
        chunk_len=args.chunk_len
    )
    
    # 执行批量推理
  
    try:
        reset_print_stream()
        answer, state = pipeline.gen_bsz(
            ctx_list, 
            state=states_value, 
            token_count=args.token_count, 
            args=pipeline_args,
            callback=callback
        )
    finally:
        if hasattr(print_stream, 'total_lines'):
            # 确保最后光标在所有输出的下方
            print('\n' * (print_stream.total_lines - 1))
        pipeline.reset_cache()
    
    
    return answer, state

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description='RWKV Batch Inference')
        
        # 环境设置
        parser.add_argument('--jit_on', type=str, default='1', help='JIT compilation switch')
        parser.add_argument('--cuda_on', type=str, default='1', help='CUDA switch')
        parser.add_argument('--fla_on', type=str, default='0', help='FLA switch')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use')
        
        # 模型和状态设置
        parser.add_argument('--base_model', type=str, 
                           default='/home/rwkv/Peter/model/base/v3-7b-industry-instruct-1024.pth',
                           help='Path to base model')
        parser.add_argument('--state_file', type=str,
                           default='/home/rwkv/Peter/model/state/entityet/v3-7b-instruct-entitieset-1024-state.pth',
                           help='Path to state file')
        
        # 批处理设置
        parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
        parser.add_argument('--token_count', type=int, default=500, help='Number of tokens to generate')
        
        # 采样参数
        parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
        parser.add_argument('--top_p', type=float, default=0, help='Top-p sampling threshold')
        parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling threshold')
        parser.add_argument('--alpha_frequency', type=float, default=0, help='Frequency penalty')
        parser.add_argument('--alpha_presence', type=float, default=0, help='Presence penalty')
        parser.add_argument('--alpha_decay', type=float, default=0, help='Decay factor')
        parser.add_argument('--chunk_len', type=int, default=256, help='Chunk length for processing')
        
        # 特殊token设置
        parser.add_argument('--token_ban', type=int, nargs='+', default=[0], help='Banned tokens')
        parser.add_argument('--token_stop', type=int, nargs='+', default=[24], help='Stop tokens')
        parser.add_argument('--vocab_size', type=int, default=65536, help='Vocabulary size')

        parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
        parser.add_argument('--output_mode', type=str, choices=['console', 'file'], 
                        default='console', help='Output mode')
        
        return parser.parse_args()

    def main():
        # 解析命令行参数
        args = parse_args()
        
        instruction = '请从输入文本中识别实体，对每个实体提取：1)实体名称 2)实体类型(从给定类型列表中选择) 3)实体的详细描述,包括属性和活动。按格式输出：(entity,实体名称,实体类型,描述)😺.'
        # 准备输入数据
        input_text1 = '本书主角，穿越者，带着成年人的记忆从地球转生到斗气大陆。 [2]本为天才少年，但从十一岁那年开始连续三年多莫名其妙地退化成斗之气三段，从此逐渐沦为遭人白眼的废柴。之后得知原因竟是有一个神秘的灵魂"药老"藏在萧炎母亲的遗物戒指中不断吸收他的斗之气，在药老停止吸收斗之气并答应帮他重展天资后，一年时间内突破至斗之气九段，震惊全城'
        input_text2 ='男主角萧炎的老师。人称药老、药尊者，后称药圣，拥有"骨灵冷火"（后赠与萧炎）。星陨阁极少露面的阁主（副阁主为至交好友风尊者-风闲），初为九转斗尊巅峰强者，九品炼药宗师。因遭叛徒韩枫出卖而落难成为灵魂状态，潜伏于一个戒指中，后戒指辗转落入萧炎之手，在吸收萧炎三年斗之气后恢复意识。'  # 你的输入文本
        entities_types = '角色,导师,修炼体系,目标,性格特点'
        
        ctx1 = f'Instruction: {instruction}\n\nInput: 文本: {input_text1}\n实体类型: {entities_types}\n\nResponse:'
        ctx2 = f'Instruction: {instruction}\n\nInput: 文本: {input_text2}\n实体类型: {entities_types}\n\nResponse:'
        
        print(len(instruction), len(input_text1), len(input_text2))
        ctx_list = [ctx1,ctx2]
        
        
        # 执行批量推理
        # 设置输出回调
        if args.output_mode == 'file':
            writer = StreamWriter(len(ctx_list), args.output_dir)
            callback = writer.callback
        else:
            callback = print_stream
        
        try:
            # 执行批量推理
            answer, state = batch_inference(ctx_list, args, callback=callback)
            
            # 打印最终结果
            print(f"\n{Fore.CYAN}Final outputs:{Style.RESET_ALL}")
            for i, ans in answer.items():
                print(f"{Fore.GREEN}Batch {i}:{Style.RESET_ALL}\n{ans}\n")
                
        finally:
            # 如果使用文件输出，确保关闭文件
            if args.output_mode == 'file':
                writer.close()

    main()

