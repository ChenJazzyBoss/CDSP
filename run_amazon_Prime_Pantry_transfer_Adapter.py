import torch.optim as optim  # 导入PyTorch优化器
import re  # 正则表达式库
from pathlib import Path  # 路径处理库
from torch.utils.data import DataLoader  # 数据加载器
import numpy as np  # 科学计算库
# 导入各种预训练模型及其tokenizer和配置
from transformers import BertModel, BertTokenizer, BertConfig, \
    RobertaTokenizer, RobertaModel, RobertaConfig, \
    DebertaTokenizer, DebertaModel, DebertaConfig, \
    DistilBertTokenizer, DistilBertModel, DistilBertConfig, \
    GPT2Tokenizer, OPTModel, OPTConfig

from parameters import parse_args  # 导入参数解析函数
# 导入自定义模型和编码器
from model.model_mmd_ada import Model2_transfer, Bert_Encoder
# 导入数据处理相关的函数
from data_utils import read_news, read_news_bert, get_doc_input_bert, get_id_embeddings, \
    read_behaviors, BuildTrainDataset, eval_model_amazon, eval_model, eval_model_step2, get_item_embeddings, \
    get_item_embeddings_llm, get_item_word_embs, get_item_word_embs_llm, get_item_embeddings_llm_4
# 导入亚马逊Pantry数据集的处理函数
from data_utils import read_news_bert_amazon_pantry, read_behaviors_amazon_pantry
from data_utils.utils import *  # 导入各种工具函数
import random  # 随机数生成库

# PyTorch分布式训练相关组件
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.init import xavier_normal_  # 权重初始化函数
import gc  # 垃圾回收
import joblib  # 用于序列化和持久化

# 设置tokenizer并行处理为false，避免潜在的冲突
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 数据集名称
datasets = "Prime_Pantry"
# LLM生成的嵌入文件名
llm_embedding = "Prime_Pantry_llm2vec.pt"
early_stop = 50


def train(args, use_modal, local_rank):
    """
    主训练函数

    参数:
    - args: 命令行参数
    - use_modal: 是否使用模态信息（文本等）
    - local_rank: 当前进程的本地排名（分布式训练使用）
    """
    # global users_train, item_word_embs  # 全局变量声明
    #
    # 如果使用模态信息（文本等）
    # global item_num, users_train, item_word_embs
    # if use_modal:
    #     根据选择的预训练模型类型，加载相应的tokenizer和模型
    #     if 'roberta' in args.bert_model_load:
    #         Log_file.info('load roberta model...')
    #         bert_model_load = '../../pretrained_models/' + args.bert_model_load
    #         tokenizer = RobertaTokenizer.from_pretrained(bert_model_load)
    #         config = RobertaConfig.from_pretrained(bert_model_load, output_hidden_states=True)
    #         bert_model = RobertaModel.from_pretrained(bert_model_load, config=config)
    #         # 根据模型大小设置词嵌入维度
    #         if 'base' in args.bert_model_load:
    #             args.word_embedding_dim = 768
    #         if 'large' in args.bert_model_load:
    #             args.word_embedding_dim = 1024
    #     elif 'opt' in args.bert_model_load:
    #         Log_file.info('load opt model...')
    #         bert_model_load = '../../pretrained_models/' + args.bert_model_load
    #         tokenizer = GPT2Tokenizer.from_pretrained(bert_model_load)
    #         config = OPTConfig.from_pretrained(bert_model_load, output_hidden_states=True)
    #         bert_model = OPTModel.from_pretrained(bert_model_load, config=config)
    #
    #     # 上面的代码都没用，直接使用的是llm的嵌入
    #     el
    #     if 'llm' in args.bert_model_load:
    #         Log_file.info('load llm2vec...')
    #         args.word_embedding_dim = 4096  # LLM的词嵌入维度设为4096
    #
    #     读取物品数据（商品信息）
    #     Log_file.info('read news...')
    #     # 物品路径
    #     items_path = os.path.join(args.root_data_dir, args.dataset, args.news)
    #     Log_file.info(f'items_path file path: {items_path}')
    #     before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name = read_news_bert_amazon_pantry(
    #         items_path, args)
    #
    #     # 读取用户行为数据
    #     Log_file.info('read behaviors...')
    #     # 行为的路径
    #     behaviors_path = os.path.join(args.root_data_dir, args.dataset, args.behaviors)
    #     Log_file.info(f'behaviors_path file path: {behaviors_path}')
    #     item_num, item_id_to_dic, users_train, users_valid, users_test, \
    #         users_history_for_valid, users_history_for_test, item_name_to_id = \
    #         read_behaviors_amazon_pantry(behaviors_path,
    #                                      before_item_id_to_dic,
    #                                      before_item_name_to_id, before_item_id_to_name,
    #                                      args.max_seq_len, args.min_seq_len, Log_file)
    #     Log_file.info('Finish reading behaviors')
    #
    #     # 加载LLM生成的物品文本嵌入
    #     item_word_embs = torch.load(f'./dataset/{datasets}/{llm_embedding}')
    #     item_word_embs = torch.tensor(item_word_embs, dtype=torch.float32)
    #
    #     Log_file.info('Finish reading item embeddings')
    Log_file.info('read news...')
    # 物品路径
    items_path = os.path.join(args.root_data_dir, args.dataset, args.news)
    Log_file.info(f'items_path file path: {items_path}')
    before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name = read_news_bert_amazon_pantry(
        items_path, args)

    # 读取用户行为数据
    Log_file.info('read behaviors...')
    # 行为的路径
    behaviors_path = os.path.join(args.root_data_dir, args.dataset, args.behaviors)
    Log_file.info(f'behaviors_path file path: {behaviors_path}')
    item_num, item_id_to_dic, users_train, users_valid, users_test, \
        users_history_for_valid, users_history_for_test, item_name_to_id = \
        read_behaviors_amazon_pantry(behaviors_path,
                                     before_item_id_to_dic,
                                     before_item_name_to_id, before_item_id_to_name,
                                     args.max_seq_len, args.min_seq_len, Log_file)
    Log_file.info('Finish reading behaviors')

    # 加载LLM生成的物品文本嵌入
    item_word_embs = torch.load(f'./dataset/{datasets}/{llm_embedding}')
    item_word_embs = torch.tensor(item_word_embs, dtype=torch.float32)

    Log_file.info('Finish reading item embeddings')
    # 构建训练数据集
    Log_file.info(f'物品总数:{item_num}')
    Log_file.info('build dataset...')
    # item_num = 8347  # 物品总数
    train_dataset = BuildTrainDataset(u2seq=users_train, item_content=item_word_embs, item_num=item_num,
                                      max_seq_len=args.max_seq_len, use_modal=use_modal)
    Log_file.info('build dataset done...')

    # 记录训练用户数量并释放内存
    len_users_train = len(users_train)
    del users_train
    gc.collect()  # 强制垃圾回收

    # 构建分布式采样器
    Log_file.info('build DDP sampler...')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    Log_file.info('before seed')

    # 为每个worker设置不同的随机种子，保证多进程采样的随机性和独立性
    def worker_init_reset_seed(worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id + dist.get_rank()
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    # 构建数据加载器
    Log_file.info('build dataloader...')
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                          worker_init_fn=worker_init_reset_seed, pin_memory=False, sampler=train_sampler)

    # 构建模型
    Log_file.info('build model...')
    model = Model2_transfer(args, item_num, use_modal).to(local_rank)
    # 将模型中的BatchNorm层转换为SyncBatchNorm，适用于分布式训练
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    # 加载检查点（如果有的话）
    if 'None' not in args.load_ckpt_name:
        Log_file.info('load ckpt if not None...')
        ckpt_path = get_checkpoint(item_emb_path, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        Log_file.info('load checkpoint...')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        Log_file.info(f"Model loaded from {ckpt_path}")
        # 从检查点名称中提取起始轮次
        start_epoch = int(re.split(r'[._-]', args.load_ckpt_name)[1])
        # 恢复随机数状态
        torch.set_rng_state(checkpoint['rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        is_early_stop = True
        model.freeze()  # 冻结模型参数
    else:
        # 如果没有检查点，从头开始训练
        checkpoint = None
        ckpt_path = None
        start_epoch = 0
        is_early_stop = True

    # 将模型包装为DistributedDataParallel模型，用于分布式训练
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    # 定义优化器
    optimizer = optim.AdamW(model.module.parameters(), lr=args.lr, weight_decay=args.l2_weight)

    # 打印模型参数总数和可训练参数数量
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Log_file.info("##### total_num {} #####".format(total_num))
    Log_file.info("##### trainable_num {} #####".format(trainable_num))

    Log_file.info('\n')
    Log_file.info('Training...')
    next_set_start_time = time.time()
    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, early_stop_count = 0, 0

    # 设置日志和评估的步骤间隔
    steps_for_log, steps_for_eval = para_and_log(model, len_users_train, args.batch_size, Log_file,
                                                 logging_num=args.logging_num, testing_num=args.testing_num)

    # 混合精度训练的缩放器
    scaler = torch.cuda.amp.GradScaler()
    if 'None' not in args.load_ckpt_name:
        # 如果有检查点，加载缩放器状态
        scaler.load_state_dict(checkpoint["scaler_state"])
        Log_file.info(f"scaler loaded from {ckpt_path}")

    Log_screen.info('{} train start'.format(args.label_screen))
    # 开始训练循环
    for ep in range(args.epoch):
        now_epoch = start_epoch + ep + 1
        Log_file.info('\n')
        Log_file.info('epoch {} start'.format(now_epoch))
        Log_file.info('')
        loss, batch_index, need_break = 0.0, 1, False
        model.train()  # 设置模型为训练模式
        train_dl.sampler.set_epoch(now_epoch)  # 设置采样器的epoch，确保每个epoch采样不同

        # 遍历数据加载器中的每个批次
        for data in train_dl:
            sample_items_id, sample_items_content, log_mask = data
            # 将数据移到对应的设备上
            sample_items_id, sample_items_content, log_mask = \
                sample_items_id.to(local_rank), sample_items_content.to(local_rank), log_mask.to(local_rank)

            # 如果使用模态信息，调整内容的形状
            if use_modal:
                sample_items_content = sample_items_content.view(-1, sample_items_content.size(-1))
            sample_items_id = sample_items_id.view(-1)

            # 优化器梯度清零
            optimizer.zero_grad()
            # 使用混合精度训练
            with torch.amp.autocast(device_type='cuda'):
                bz_loss = model(sample_items_id, sample_items_content, log_mask, local_rank)
                loss += bz_loss.data.float()
            # 缩放损失值以避免数值精度问题
            scaler.scale(bz_loss).backward()
            # 执行优化步骤
            scaler.step(optimizer)
            scaler.update()

            # 如果损失为NaN，停止训练
            if torch.isnan(loss.data):
                need_break = True
                break

            # 定期记录训练损失
            if batch_index % steps_for_log == 0:
                Log_file.info('cnt: {}, Ed: {}, batch loss: {:.5f}, sum loss: {:.5f}'.format(
                    batch_index, batch_index * args.batch_size, loss.data / batch_index, loss.data))
            batch_index += 1

        # 每隔一定轮次进行评估
        if not need_break and now_epoch % 1 == 0:
            Log_file.info('')
            # 运行评估，获取评估结果和早停相关信息
            max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save = \
                run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
                         model, item_word_embs, users_history_for_valid, users_valid, args.batch_size, item_num,
                         use_modal,
                         args.mode, is_early_stop, local_rank)
            model.train()  # 恢复训练模式
            # 如果需要保存模型且是主进程
            if need_save and dist.get_rank() == 0:
                save_model(now_epoch, model, model_dir, optimizer,
                           torch.get_rng_state(), torch.cuda.get_rng_state(), scaler, Log_file)
        Log_file.info('')
        # 报告当前轮次的训练时间
        next_set_start_time = report_time_train(batch_index, now_epoch, loss, next_set_start_time, start_time, Log_file)
        Log_screen.info('{} training: epoch {}/{}'.format(args.label_screen, now_epoch, args.epoch))
        # 如果需要提前结束训练
        if need_break:
            break

    # 训练结束，保存最终模型（仅在主进程上）
    if dist.get_rank() == 0:
        save_model(now_epoch, model, model_dir, optimizer,
                   torch.get_rng_state(), torch.cuda.get_rng_state(), scaler, Log_file)

    # 打印最终评估结果和训练信息
    Log_file.info('\n')
    Log_file.info('%' * 90)
    Log_file.info(' max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value * 100, max_epoch))
    Log_file.info(' early stop in epoch {}'.format(early_stop_epoch))
    Log_file.info('the End')
    Log_screen.info('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))

    # 获取物品嵌入并进行最终测试评估
    item_embeddings = get_item_embeddings_llm_4(model, item_word_embs, args.batch_size, args, use_modal, local_rank)
    valid_Hit10 = eval_model_step2(model, users_history_for_test, users_test, item_embeddings, args.batch_size, args,
                                   item_num, Log_file, args.mode, local_rank)


def run_eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count,
             model, item_word_embs, user_history, users_eval, batch_size, item_num, use_modal,
             mode, is_early_stop, local_rank):
    """
    运行评估函数

    参数:
    - now_epoch: 当前轮次
    - max_epoch: 最佳性能轮次
    - early_stop_epoch: 早停轮次
    - max_eval_value: 最佳评估指标值
    - early_stop_count: 早停计数器
    - model: 模型
    - item_word_embs: 物品文本嵌入
    - user_history: 用户历史行为
    - users_eval: 评估用户集
    - batch_size: 批次大小
    - item_num: 物品数量
    - use_modal: 是否使用模态信息
    - mode: 训练模式
    - is_early_stop: 是否早停
    - local_rank: 本地进程排名

    返回:
    - 更新后的最佳评估值、最佳轮次、早停轮次、早停计数、是否需要提前结束训练、是否需要保存模型
    """
    eval_start_time = time.time()
    Log_file.info('Validating...')

    # 获取物品嵌入
    item_embeddings = get_item_embeddings_llm_4(model, item_word_embs, batch_size, args, use_modal, local_rank)
    # 评估模型性能（Hit@10指标）
    valid_Hit10 = eval_model_step2(model, user_history, users_eval, item_embeddings, batch_size, args,
                                   item_num, Log_file, mode, local_rank)
    # 报告评估时间
    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')

    need_break = False
    need_save = False

    # 如果当前评估结果优于历史最佳，更新最佳值和相关计数
    if valid_Hit10 > max_eval_value:
        max_eval_value = valid_Hit10
        max_epoch = now_epoch
        early_stop_count = 0
        need_save = True  # 标记需要保存模型
    else:
        # 否则增加早停计数
        early_stop_count += 1
        # 如果连续20次评估没有提升，触发早停机制
        if early_stop_count > early_stop:
            if is_early_stop:
                need_break = True  # 标记需要提前结束训练
            early_stop_epoch = now_epoch

    return max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break, need_save


def setup_seed(seed):
    """
    设置随机种子，确保实验的可重复性

    参数:
    - seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 设置CUDNN为确定性模式，可能会影响性能，但保证结果可复现
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
    # 获取可用GPU数量
    gpus = torch.cuda.device_count()
    early_stop = args.early_stop

    item_emb = f"Tower_1_{args.embedding_dim}"
    # 根据物品塔类型决定是否使用模态信息
    if 'modal' in args.item_tower:
        # 使用文本等模态信息
        is_use_modal = True
        # model_load = '/'
        # 设置目录标签和日志参数
        tower_name = f"Tower_{args.tower}_{args.embedding_dim}"
        dir_label = os.path.join(args.dataset, tower_name)
        # 构建包含其他参数的子目录或文件名
        log_paras = f"bs{args.batch_size}_lr{args.lr}_modnn{args.mo_dnn_layers}_dnn{args.dnn_layers}"

    else:
        # 不使用模态信息，仅使用ID
        is_use_modal = False
        # model_load = '/'
        # 设置目录标签和日志参数
        tower_name = f"Tower_{args.tower}_{args.embedding_dim}"
        dir_label = os.path.join(args.dataset, tower_name)
        # 构建包含其他参数的子目录或文件名
        log_paras = f"bs{args.batch_size}_lr{args.lr}_modnn{args.mo_dnn_layers}_dnn{args.dnn_layers}"

    # 项目的ID嵌入
    item_emb_path = os.path.join('./checkpoint', args.dataset, item_emb)
    # 设置模型保存路径
    model_dir = os.path.join('./checkpoint', dir_label)
    # 生成时间标记，用于日志文件命名
    time_run = time.strftime('-%Y%m%d-%H%M%S', time.localtime())
    args.label_screen = args.label_screen + time_run

    # 设置日志记录器
    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank(), args.behaviors)
    Log_file.info(args)
    # 创建模型保存目录（如果不存在）
    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    # 记录开始时间
    start_time = time.time()
    # 如果是训练模式，启动训练
    if 'train' in args.mode:
        print(local_rank)
        train(args, is_use_modal, local_rank)
    # 记录结束时间并计算总耗时
    end_time = time.time()
    hour, minutes, seconds = get_time(start_time, end_time)
    Log_file.info("##### (time) all: {} hours {} minutes {} seconds #####".format(hour, minutes, seconds))
    # 打印gamma参数和冻结状态
    print(args.gamma)
    Log_file.info("##### freeze: {} gamma: {}".format(args.freeze, args.gamma))
