import torch.optim as optim
import re
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig, \
    RobertaTokenizer, RobertaModel, RobertaConfig, \
    DebertaTokenizer, DebertaModel, DebertaConfig, \
    DistilBertTokenizer, DistilBertModel, DistilBertConfig, \
    GPT2Tokenizer, OPTModel, OPTConfig

from parameters import parse_args
from model.model_mmd_ada import Model_new3_3, Model2, Model2_align, Model2_transfer, Bert_Encoder
from data_utils import eval_model_2_3tower_amazon_pantry, BuildTrainDataset_new_amazon_pantry, \
    get_item_embeddings_llm_3tower, eval_model_2_3tower_amazon, eval_model_2_2_amazon, BuildTrainDataset_new_amazon_ele, \
    read_news, read_news_bert, get_doc_input_bert, get_id_embeddings_amazon, \
    read_behaviors, BuildTrainDataset, eval_model_amazon, eval_model, eval_model_step2, get_item_embeddings, \
    get_item_embeddings_llm, get_item_word_embs, get_item_word_embs_llm, get_item_embeddings_llm_4
from data_utils import read_news_bert_amazon_pantry, read_behaviors_amazon_pantry
from data_utils.utils import *
import random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.init import xavier_normal_
import gc
import joblib

# 禁用tokenizers并行处理，避免与分布式训练冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 定义数据集名称和预训练的LLM嵌入文件名
datasets = "Prime_Pantry"
llm_embedding = "Prime_Pantry_llm2vec.pt"
early_stop = 50


def train(args, use_modal, local_rank):
    """
    主训练函数，实现了基于Transformer的推荐系统模型训练过程

    参数:
        args: 命令行解析的参数对象
        use_modal: 是否使用模态信息（如文本特征）
        local_rank: 当前进程的本地GPU序号
    """
    global item_num, users_train, item_word_embs
    if use_modal:
        # 根据命令行参数加载不同的预训练语言模型
        if 'roberta' in args.bert_model_load:
            Log_file.info('load roberta model...')
            bert_model_load = '../../pretrained_models/' + args.bert_model_load
            tokenizer = RobertaTokenizer.from_pretrained(bert_model_load)
            config = RobertaConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = RobertaModel.from_pretrained(bert_model_load, config=config)
            # 设置词嵌入维度
            if 'base' in args.bert_model_load:
                args.word_embedding_dim = 768
            if 'large' in args.bert_model_load:
                args.word_embedding_dim = 1024
        elif 'opt' in args.bert_model_load:
            Log_file.info('load opt model...')
            bert_model_load = '../../pretrained_models/' + args.bert_model_load
            tokenizer = GPT2Tokenizer.from_pretrained(bert_model_load)
            config = OPTConfig.from_pretrained(bert_model_load, output_hidden_states=True)
            bert_model = OPTModel.from_pretrained(bert_model_load, config=config)
        elif 'llm' in args.bert_model_load:
            Log_file.info('load llm2vec...')
            args.word_embedding_dim = 4096  # LLM嵌入维度为4096

        # 读取商品数据
        Log_file.info('read news...')
        before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name = read_news_bert_amazon_pantry(
            os.path.join(args.root_data_dir, args.dataset, args.news), args)

        # 读取用户行为数据
        Log_file.info('read behaviors...')
        item_num, item_id_to_dic, users_train, users_valid, users_test, \
            users_history_for_valid, users_history_for_test, item_name_to_id = \
            read_behaviors_amazon_pantry(os.path.join(args.root_data_dir, args.dataset, args.behaviors),
                                         before_item_id_to_dic,
                                         before_item_name_to_id, before_item_id_to_name,
                                         args.max_seq_len, args.min_seq_len, Log_file)
        Log_file.info('Finish reading behaviors')

        # 加载预训练的商品嵌入向量
        item_word_embs = torch.load(f'./dataset/{datasets}/{llm_embedding}')
        item_word_embs = torch.tensor(item_word_embs, dtype=torch.float32)
        Log_file.info('Finish reading item embeddings')

    Log_file.info('build dataset...')

    # 设置商品数量
    # item_num = 8347
    # 构建训练数据集对象
    train_dataset = BuildTrainDataset_new_amazon_pantry(u2seq=users_train, item_content=item_word_embs,
                                                        item_num=item_num,
                                                        max_seq_len=args.max_seq_len, use_modal=use_modal)
    Log_file.info('build dataset done...')
    len_users_train = len(users_train)
    del users_train  # 释放内存
    gc.collect()

    # 构建分布式采样器
    Log_file.info('build DDP sampler...')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    Log_file.info('before seed')

    # 定义工作线程的随机种子初始化函数，确保分布式训练中的随机性是可复现的
    def worker_init_reset_seed(worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id + dist.get_rank()
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    # 构建数据加载器
    Log_file.info('build dataloader...')
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                          worker_init_fn=worker_init_reset_seed, pin_memory=True, sampler=train_sampler)

    # 构建模型
    Log_file.info('build model...')
    model = Model_new3_3(args, item_num, use_modal).to(local_rank)
    # 将模型中的普通BatchNorm转换为同步BatchNorm，以便在分布式训练中正确计算批归一化统计量
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    if use_modal:
        # 输出模型中的关键组件信息
        Log_file.info(model.turn_dim1)
        Log_file.info(model.fc)
        Log_file.info(model.mlp_layers)

    # 如果指定了加载检查点，则从检查点恢复模型状态
    if 'None' not in args.load_ckpt_name:
        Log_file.info('load ckpt if not None...')
        ckpt_path = get_checkpoint(item_emb_path, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        Log_file.info('load checkpoint...')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        Log_file.info(f"Model loaded from {ckpt_path}")
        # 从检查点名称中解析起始轮次
        start_epoch = int(re.split(r'[._-]', args.load_ckpt_name)[1])
        # 恢复随机数生成器状态，确保训练的随机性是可复现的
        try:
            # 尝试恢复 CUDA 随机数状态
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
            print("✅ CUDA RNG 状态恢复成功")

        except RuntimeError as e:
            # 如果失败，优雅地处理错误
            print(f"⚠️ 跳过 CUDA RNG 状态恢复: {e}")
            print("💡 这不会影响模型权重，只是随机数序列可能不同")

        # torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        is_early_stop = True
        model.freeze6()  # 冻结部分模型参数
    else:
        checkpoint = None  # 新训练
        ckpt_path = None  # 新训练
        start_epoch = 0
        is_early_stop = True

    # 将模型包装为分布式数据并行模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 将模型参数分为两组：1) alpha和beta参数，2) 其他参数
    # 这样可以为不同组的参数设置不同的学习率和权重衰减
    w_params = [param for name, param in model.named_parameters() if 'alpha' in name or 'beta' in name]
    b_params = [param for name, param in model.named_parameters() if not 'alpha' in name and not 'beta' in name]
    optimizer = optim.AdamW(
        [{'params': w_params, 'lr': args.lr}, {'params': b_params, 'lr': args.lr, 'weight_decay': args.l2_weight}])

    # 输出模型参数信息
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Log_file.info("##### total_num {} #####".format(total_num))
    Log_file.info("##### trainable_num {} #####".format(trainable_num))

    # 开始训练过程
    Log_file.info('\n')
    Log_file.info('Training...')
    next_set_start_time = time.time()
    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, early_stop_count = 0, 0
    # 计算日志记录和评估的步骤间隔
    steps_for_log, steps_for_eval = para_and_log(model, len_users_train, args.batch_size, Log_file,
                                                 logging_num=args.logging_num, testing_num=args.testing_num)
    # 创建混合精度训练的梯度缩放器
    scaler = torch.cuda.amp.GradScaler()
    if 'None' not in args.load_ckpt_name:
        scaler.load_state_dict(checkpoint["scaler_state"])
        Log_file.info(f"scaler loaded from {ckpt_path}")

    # 在屏幕上输出训练开始信息
    Log_screen.info('{} train start'.format(args.label_screen))
    # 开始训练循环
    for ep in range(args.epoch):
        now_epoch = start_epoch + ep + 1
        Log_file.info('\n')
        Log_file.info('epoch {} start'.format(now_epoch))
        Log_file.info('')
        loss, batch_index, need_break = 0.0, 1, False
        model.train()  # 设置模型为训练模式
        train_dl.sampler.set_epoch(now_epoch)  # 设置采样器的轮次，确保不同轮次的数据顺序不同

        # 遍历数据加载器中的每个批次
        for data in train_dl:
            sample_items_id, sample_items_content, log_mask, bin_pos, bin_neg = data
            # 将数据移动到GPU
            sample_items_id, sample_items_content, log_mask, bin_pos, bin_neg = \
                sample_items_id.to(local_rank), sample_items_content.to(local_rank), log_mask.to(
                    local_rank), bin_pos.to(local_rank), bin_neg.to(local_rank)

            # 重塑输入形状
            if use_modal:
                sample_items_content = sample_items_content.view(-1, sample_items_content.size(-1))
            sample_items_id = sample_items_id.view(-1)

            # 清除优化器中的梯度
            optimizer.zero_grad()
            # 使用混合精度训练
            with torch.amp.autocast(device_type='cuda'):
                # 前向传播计算损失
                bz_loss = model(sample_items_id, sample_items_content, log_mask, bin_pos, bin_neg, local_rank)
                loss += bz_loss.data.float()
            # 使用梯度缩放进行反向传播和优化器更新
            scaler.scale(bz_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 检查损失是否为NaN，如果是则中断训练
            if torch.isnan(loss.data):
                need_break = True
                break

            # 记录训练日志
            if batch_index % steps_for_log == 0:
                Log_file.info('cnt: {}, Ed: {}, batch loss: {:.5f}, sum loss: {:.5f}'.format(
                    batch_index, batch_index * args.batch_size, loss.data / batch_index, loss.data))
            batch_index += 1

        # 定期评估模型性能
        if not need_break and now_epoch % 1 == 0:
            Log_file.info('')
            # 运行验证集评估，并根据结果决定是否提前停止训练
            max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save = \
                run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                         model, item_word_embs, users_history_for_valid, users_valid, args.batch_size, item_num,
                         use_modal,
                         args.mode, is_early_stop, local_rank)
            model.train()  # 评估后将模型设回训练模式
            # 如果模型性能提升，则保存模型
            if need_save and dist.get_rank() == 0:
                save_model(now_epoch, model, model_dir, optimizer,
                           torch.get_rng_state(), torch.cuda.get_rng_state(), scaler, Log_file)
        Log_file.info('')
        # 输出本轮训练的时间信息
        next_set_start_time = report_time_train(batch_index, now_epoch, loss, next_set_start_time, start_time, Log_file)
        Log_screen.info('{} training: epoch {}/{}'.format(args.label_screen, now_epoch, args.epoch))
        if need_break:
            break

    # 在训练结束时保存最终模型
    if dist.get_rank() == 0:
        save_model(now_epoch, model, model_dir, optimizer,
                   torch.get_rng_state(), torch.cuda.get_rng_state(), scaler, Log_file)

    # 输出训练结果摘要
    Log_file.info('\n')
    Log_file.info('%' * 90)
    Log_file.info(' max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value * 100, max_epoch))
    Log_file.info(' early stop in epoch {}'.format(early_stop_epoch))
    Log_file.info('the End')
    Log_screen.info('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))
    Log_file.info('gamma2 {}'.format(args.gamma2))
    Log_file.info('lr {}'.format(args.lr))

    # 在测试集上进行最终评估
    item_embeddings3, item_embeddings, id_embs = get_item_embeddings_llm_3tower(model, item_word_embs, args.batch_size,
                                                                                args, use_modal, local_rank)
    valid_Hit10 = eval_model_2_3tower_amazon_pantry(10, model, users_history_for_test, users_test, item_embeddings3,
                                                    item_embeddings, id_embs, 512, args,
                                                    item_num, Log_file, args.mode, local_rank)


def run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
             model, item_word_embs, user_history, users_eval, batch_size, item_num, use_modal,
             mode, is_early_stop, local_rank):
    """
    在验证集上评估模型性能

    参数:
        now_epoch: 当前训练轮次
        max_epoch: 最佳性能轮次
        early_stop_epoch: 提前停止的轮次
        max_eval_value: 最佳验证集性能
        early_stop_count: 性能未提升的连续轮次计数
        model: 要评估的模型
        item_word_embs: 商品文本嵌入
        user_history: 用户历史行为
        users_eval: 用于评估的用户数据
        batch_size: 批次大小
        item_num: 商品数量
        use_modal: 是否使用模态信息
        mode: 运行模式
        is_early_stop: 是否启用提前停止
        local_rank: 当前进程的GPU序号

    返回:
        max_eval_value: 更新后的最佳验证集性能
        max_epoch: 更新后的最佳性能轮次
        early_stop_epoch: 更新后的提前停止轮次
        early_stop_count: 更新后的性能未提升计数
        need_break: 是否需要中断训练
        need_save: 是否需要保存模型
    """
    eval_start_time = time.time()
    Log_file.info('Validating...')
    # 获取商品嵌入向量
    item_embeddings3, item_embeddings, id_embs = get_item_embeddings_llm_3tower(model, item_word_embs, batch_size, args,
                                                                                use_modal, local_rank)
    # 计算Hit@10指标
    valid_Hit10 = eval_model_2_3tower_amazon_pantry(10, model, user_history, users_eval, item_embeddings3,
                                                    item_embeddings, id_embs, 512, args, item_num, Log_file, mode,
                                                    local_rank)
    # 记录评估时间
    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')

    need_break = False
    need_save = False
    # 判断模型性能是否提升
    if valid_Hit10 > max_eval_value:
        max_eval_value = valid_Hit10
        max_epoch = now_epoch
        early_stop_count = 0
        need_save = True
    else:
        early_stop_count += 1
        # 如果连续20轮性能未提升，则考虑提前停止
        if early_stop_count > early_stop:
            if is_early_stop:
                need_break = True
            early_stop_epoch = now_epoch
    return max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save


def setup_seed(seed):
    """
    设置随机种子以确保实验可复现性

    参数:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 初始化分布式训练环境
    dist.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    # 设置随机种子
    setup_seed(12345)
    gpus = torch.cuda.device_count()
    early_stop = args.early_stop

    item_emb = f"Tower_2_{args.embedding_dim}"
    # 根据模型架构选择确定是否使用模态信息
    if 'modal' in args.item_tower:
        is_use_modal = True
        model_load = '/'
        flag = 0.0001
        # 设置目录标签和日志参数
        tower_name = f"Tower_{args.tower}_{args.embedding_dim}"
        dir_label = os.path.join(args.dataset, tower_name)
        # 构建包含其他参数的子目录或文件名
        log_paras = f"bs{args.batch_size}_lr{args.lr}_modnn{args.mo_dnn_layers}_dnn{args.dnn_layers}"
    else:
        is_use_modal = False
        # 设置目录标签和日志参数
        tower_name = f"Tower_{args.tower}_{args.embedding_dim}"
        dir_label = os.path.join(args.dataset, tower_name)
        # 构建包含其他参数的子目录或文件名
        log_paras = f"bs{args.batch_size}_lr{args.lr}_modnn{args.mo_dnn_layers}_dnn{args.dnn_layers}"

    # 项目的ID嵌入
    item_emb_path = os.path.join('./checkpoint', args.dataset, item_emb)
    # 设置模型保存路径
    model_dir = os.path.join('./checkpoint', dir_label)
    # 生成时间戳，用于标记日志文件
    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
    args.label_screen = args.label_screen + time_run

    # 设置日志记录器
    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank(), args.behaviors)
    Log_file.info(args)
    # 创建模型保存目录
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    # 记录开始时间
    start_time = time.time()
    # 根据模式运行训练或评估
    if 'train' in args.mode:
        print(local_rank)
        train(args, is_use_modal, local_rank)
    # 记录结束时间并输出总耗时
    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    Log_file.info("##### (time) all: {} hours {} minutes {} seconds #####".format(hour, minu, secon))
    print(args.gamma)
    Log_file.info("##### freeze: {} gamma: {}".format(args.freeze, args.gamma))
