import configargparse
import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True,help="Dir of Source domain")
    parser.add_argument('--tgt_domain', type=str, required=True,help="Dir of Target domain")
    parser.add_argument('--num_bands',type=int, required=True,help="Number of bands in the hyperspectral image") # Modify the input of network according to the number of bands
    parser.add_argument('--num_samples', type=int, default=1500, help="Number of samples to be extracted from each class")
    parser.add_argument('--test_ratio', type=float, default=0.3, help="Ratio of test samples") # test all samples in default
    parser.add_argument('--patch_size', type=int, default=9, help="Patch size for the hyperspectral image")
    
    # training related
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=10, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=500, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=1)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    parser.add_argument('--dis_loss_weight', type=float, default=1)

    # semantic branch related
    parser.add_argument('--use_semantic_branch', type=str2bool, default=False)
    parser.add_argument('--semantic_path', type=str, default='')
    parser.add_argument('--semantic_dim', type=int, default=0)
    parser.add_argument('--shared_dim', type=int, default=128)
    parser.add_argument('--semantic_hidden_dim', type=int, default=0)
    parser.add_argument('--semantic_conf_threshold', type=float, default=0.9)
    parser.add_argument('--semantic_metric', type=str, default='cosine')
    parser.add_argument('--semantic_logit_scale', type=float, default=16.0)
    parser.add_argument('--semantic_normalize', type=str2bool, default=True)
    parser.add_argument('--semantic_loss_weight', type=float, default=1.0)
    parser.add_argument('--semantic_start_epoch', type=int, default=40)
    parser.add_argument('--semantic_ramp_end_epoch', type=int, default=55)
    parser.add_argument('--semantic_src_weight', type=float, default=1.0)
    parser.add_argument('--semantic_tgt_weight', type=float, default=1.0)
    parser.add_argument('--semantic_tgt_warmup_epochs', type=int, default=0)
    parser.add_argument('--semantic_margin_threshold', type=float, default=0.05)
    parser.add_argument('--semantic_decay_power', type=float, default=1.0)
    parser.add_argument('--semantic_tgt_ramp_power', type=float, default=1.0)
    parser.add_argument('--semantic_beta', type=float, default=0.5)
    parser.add_argument('--debug_semantic', type=str2bool, default=False)
    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=False,target= False,train=True, num_workers=args.num_workers, num_samples=args.num_samples, patch_size=args.patch_size)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, target= True,train=True, num_workers=args.num_workers, num_samples=args.num_samples, patch_size=args.patch_size)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size,infinite_data_loader=False ,target= True,train=False, num_workers=args.num_workers, test_ratio=1, patch_size=args.patch_size)
    return source_loader, target_train_loader, target_test_loader, n_class

def get_model(args):
    # If semantic branch contributes zero to total objective, fully disable it to
    # avoid unnecessary graph construction / optimizer param groups.
    semantic_branch_active = args.use_semantic_branch and (
        args.semantic_loss_weight > 0 and (args.semantic_src_weight > 0 or args.semantic_tgt_weight > 0)
    )

    if args.use_semantic_branch and not semantic_branch_active:
        print(
            '[semantic] semantic branch is requested but total semantic objective is zero; '
            'disabling semantic branch for this run.'
        )

    model = models.NewTransferNet(
        args.n_class,
        transfer_loss=args.transfer_loss,
        base_net=args.backbone,
        max_iter=args.max_iter,
        input_channels=args.num_bands,
        use_bottleneck=args.use_bottleneck,
        use_semantic_branch=semantic_branch_active,
        semantic_path=args.semantic_path,
        semantic_dim=args.semantic_dim,
        shared_dim=args.shared_dim,
        semantic_hidden_dim=args.semantic_hidden_dim,
        semantic_conf_threshold=args.semantic_conf_threshold,
        semantic_metric=args.semantic_metric,
        semantic_logit_scale=args.semantic_logit_scale,
        semantic_normalize=args.semantic_normalize,
        semantic_src_weight=args.semantic_src_weight,
        semantic_tgt_weight=args.semantic_tgt_weight,
        semantic_tgt_warmup_epochs=args.semantic_tgt_warmup_epochs,
        semantic_margin_threshold=args.semantic_margin_threshold,
        semantic_decay_power=args.semantic_decay_power,
        semantic_tgt_ramp_power=args.semantic_tgt_ramp_power,
        semantic_beta=args.semantic_beta,
        n_epoch=args.n_epoch,
        debug_semantic=args.debug_semantic,
    ).to(args.device)
    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    
    all_targets = []
    all_preds = []
    all_features = []
    
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
            features = model.get_features(data)

            all_features.extend(features.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            
    acc = 100. * correct / len_target_dataset
    
    # Calculate per-class accuracy (robust to classes with zero test samples)
    conf_mat = confusion_matrix(all_targets, all_preds, labels=np.arange(args.n_class))
    class_totals = np.sum(conf_mat, axis=1)
    per_class_acc = np.divide(
        np.diag(conf_mat),
        class_totals,
        out=np.zeros_like(class_totals, dtype=float),
        where=class_totals != 0,
    )

    # Calculate OA, AA, and Kappa
    oa = accuracy_score(all_targets, all_preds)
    valid_mask = class_totals != 0
    aa = np.mean(per_class_acc[valid_mask]) if np.any(valid_mask) else 0.0
    kappa = cohen_kappa_score(all_targets, all_preds)
    
    return acc, test_loss.avg, per_class_acc, oa, aa, kappa, all_features, all_targets

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    def get_semantic_weight(epoch, lambda_sem_max, start_epoch, ramp_end_epoch):
        if epoch < start_epoch:
            return 0.0
        if ramp_end_epoch <= start_epoch:
            return lambda_sem_max
        if epoch < ramp_end_epoch:
            return lambda_sem_max * (epoch - start_epoch) / float(ramp_end_epoch - start_epoch)
        return lambda_sem_max

    start_time = time.time()
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch 

    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    best_per_class_acc = None
    best_oa = 0
    best_aa = 0
    best_kappa = 0
    best_features = None
    best_targets = None
    stop = 0
    log = []

    # 计算最后几次迭代的准确率
    Aver_oa = 0
    Aver_aa = 0
    Aver_kappa = 0
    Aver_per_class_acc = []


    for e in range(1, args.n_epoch+1):
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_dis = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        train_loss_sem = utils.AverageMeter()
        train_loss_sem_src = utils.AverageMeter()
        train_loss_sem_tgt = utils.AverageMeter()
        train_sem_valid_ratio = utils.AverageMeter()
        train_sem_conf_ratio = utils.AverageMeter()
        train_sem_margin_ratio = utils.AverageMeter()
        train_sem_src_coarse = utils.AverageMeter()
        train_sem_src_fine = utils.AverageMeter()
        train_lambda_sem = utils.AverageMeter()
        train_lambda_sem_tgt = utils.AverageMeter()
        model.epoch_based_processing(n_batch)

        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        criterion = torch.nn.CrossEntropyLoss()
        lambda_sem = get_semantic_weight(
            e,
            args.semantic_loss_weight,
            args.semantic_start_epoch,
            args.semantic_ramp_end_epoch,
        )
        semantic_enabled = model.use_semantic_branch and (lambda_sem > 0)

        for batch_idx in range(n_batch):
            data_source, label_source = next(iter_source) # .next()
            data_target, target_label_unused = next(iter_target) # .next()
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)

            clf_loss, dis_loss, transfer_loss, semantic_metrics = model(
                data_source,
                data_target,
                label_source,
                epoch=e,
                semantic_enabled=semantic_enabled,
            )
            sem_loss_raw = semantic_metrics['sem_loss']
            sem_loss = lambda_sem * sem_loss_raw
            if model.use_semantic_branch and batch_idx == 0 and args.debug_semantic:
                print(f"[semantic][epoch {e}] src coarse logits shape: {semantic_metrics['source_coarse_sem_logits_shape']}")
                print(f"[semantic][epoch {e}] src fine logits shape: {semantic_metrics['source_fine_sem_logits_shape']}")
            loss = clf_loss + args.transfer_loss_weight * transfer_loss + args.dis_loss_weight * dis_loss
            if model.use_semantic_branch:
                loss = loss + sem_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_dis.update(dis_loss.item())
            # train_loss_class.update(class_loss.item())
            train_loss_sem.update(sem_loss.item())
            train_loss_sem_src.update(semantic_metrics['sem_src_loss'].item())
            train_loss_sem_tgt.update(semantic_metrics['sem_tgt_loss'].item())
            train_sem_valid_ratio.update(semantic_metrics['sem_valid_ratio'].item())
            train_sem_conf_ratio.update(semantic_metrics['sem_conf_pass_ratio'].item())
            train_sem_margin_ratio.update(semantic_metrics['sem_margin_pass_ratio'].item())
            train_sem_src_coarse.update(semantic_metrics['sem_src_coarse_loss'].item())
            train_sem_src_fine.update(semantic_metrics['sem_src_fine_loss'].item())
            train_lambda_sem.update(semantic_metrics['lambda_sem_t'].item())
            train_lambda_sem_tgt.update(semantic_metrics['lambda_sem_tgt'].item())
            train_loss_total.update(loss.item())

        log.append([
            train_loss_clf.avg,
            train_loss_transfer.avg,
            train_loss_dis.avg,
            train_loss_sem.avg,
            train_loss_sem_src.avg,
            train_loss_sem_tgt.avg,
            train_sem_valid_ratio.avg,
            train_sem_conf_ratio.avg,
            train_sem_margin_ratio.avg,
            train_sem_src_coarse.avg,
            train_sem_src_fine.avg,
            train_lambda_sem.avg,
            train_lambda_sem_tgt.avg,
            train_loss_total.avg,
        ])

        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, dis_loss: {:.4f}, sem_loss: {:.4f}, sem_src: {:.4f}, sem_tgt: {:.4f}, sem_ratio: {:.4f}, sem_conf: {:.4f}, sem_margin: {:.4f}, sem_src_coarse: {:.4f}, sem_src_fine: {:.4f}, lambda_sem: {:.4f}, lambda_sem_t: {:.4f}, lambda_sem_tgt: {:.4f}, total_Loss: {:.4f}'.format(
            e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_dis.avg, train_loss_sem.avg, train_loss_sem_src.avg, train_loss_sem_tgt.avg, train_sem_valid_ratio.avg, train_sem_conf_ratio.avg, train_sem_margin_ratio.avg, train_sem_src_coarse.avg, train_sem_src_fine.avg, lambda_sem, train_lambda_sem.avg, train_lambda_sem_tgt.avg, train_loss_total.avg)
        # Test
        stop += 1
        test_acc, test_loss, per_class_acc, oa, aa, kappa, all_features, all_targets = test(model, target_test_loader, args)
        info += ', test_loss {:.4f}, test_acc: {:.4f}, OA: {:.4f}, AA: {:.4f}, Kappa: {:.4f}'.format(test_loss, test_acc, oa, aa, kappa)


        if e >= args.n_epoch - 9:
            Aver_oa += oa
            Aver_aa += aa
            Aver_kappa += kappa
            Aver_per_class_acc = [x + y for x, y in zip(Aver_per_class_acc, per_class_acc)]

        np_log = np.array(log, dtype=float)
        # np.savetxt('./log/train_log.csv', np_log, delimiter=',', fmt='%.6f')
        
        if best_acc < test_acc:
            best_acc = test_acc
            best_per_class_acc = per_class_acc
            best_oa = oa
            best_aa = aa
            best_kappa = kappa
            best_features = all_features
            best_targets = all_targets
            stop = 0
        print(info)

    
        # early stopping

        if (args.data_dir).split("/")[-1] == 'Pavia':
            if best_acc > 85:
                break
        elif (args.data_dir).split("/")[-1] == 'Houston':
            if best_acc > 74:
                break
        elif (args.data_dir).split("/")[-1] == 'HyRANK':
            if best_acc > 66:
                break

        
    print('Transfer result: acc: {:.4f}, per_class_acc: {}, oa: {:.4f}, aa: {:.4f}, kappa: {:.4f}'.format(best_acc, best_per_class_acc, best_oa, best_aa, best_kappa))
    end_time = time.time()  # 记录训练结束时间
    total_time = end_time - start_time
    print(f"Total training time: {total_time} seconds")



def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)
    

if __name__ == "__main__":
    main()
