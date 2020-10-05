from data.data_pipe import de_preprocess, get_train_loader, get_val_data, get_common_val_data
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm, MetricNet
from verifacation import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import torch.nn as nn

plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz
import torchvision
from metric_learning import ArcMarginProduct, AddMarginProduct, AdaCos


def denormalize_image(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), is_tensor=True):
    max_pixel_value = 255.
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value
    denominator = np.reciprocal(std, dtype=np.float32)
    if is_tensor:
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
    img /= denominator
    img += mean
    if is_tensor:
        img = img.astype(np.uint8)
    return img


class face_learner(object):
    def __init__(self, conf, inference=False, train_transforms=None, val_transforms=None, train_loader=None):
        print(conf)
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            if conf.net_mode in ['ir', 'ir_se']:
                self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            else:
                import json
                self.model = MetricNet(model_name=conf.net_mode,
                                       pooling=conf.pooling,
                                       use_fc=True,
                                       fc_dim=conf.embedding_size,
                                       dropout=conf.last_fc_dropout,
                                       pretrained=conf.pretrained).to(conf.device)
                print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))

            if not inference:
                self.milestones = conf.milestones
                if train_loader is None:
                    self.loader, self.class_num = get_train_loader(conf, train_transforms)
                else:
                    self.loader = train_loader
                    self.class_num = conf.num_classes

                self.writer = SummaryWriter(conf.log_path)
                self.step = 0

                if conf.use_mobilfacenet or conf.net_mode in ['ir', 'ir_se']:
                    self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
                else:
                    if conf.loss_module == 'arcface':
                        self.head = ArcMarginProduct(self.model.final_in_features, self.class_num,
                                                     s=conf.s, m=conf.margin, easy_margin=False, ls_eps=conf.ls_eps).to(conf.device)
                    elif conf.loss_module == 'cosface':
                        self.head = AddMarginProduct(self.model.final_in_features, self.class_num, s=conf.s,
                                                     m=conf.margin).to(conf.device)
                    elif conf.loss_module == 'adacos':
                        self.head = AdaCos(self.model.final_in_features, self.class_num, m=conf.margin,
                                           theta_zero=conf.theta_zero).to(conf.device)
                    else:
                        self.head = nn.Linear(self.model.final_in_features, self.class_num).to(conf.device)

                print('two model heads generated')

                paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

                if conf.use_mobilfacenet:
                    params = [
                        {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                        {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                        {'params': paras_only_bn}
                    ]
                    wd = 4e-5
                else:
                    if conf.net_mode in ['ir', 'ir_se']:
                        params = [
                            {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                            {'params': paras_only_bn}
                        ]
                        wd = 5e-4
                    else:
                        params = self.model.parameters()
                        wd = conf.wd
                        # params = [
                        #     {'params': paras_wo_bn + [self.head.weight], 'weight_decay': conf.wd},  # 5e-4},
                        #     {'params': paras_only_bn}
                        # ]

                if conf.optimizer == 'sgd':
                    self.optimizer = optim.SGD(params, lr=conf.lr, momentum=conf.momentum, weight_decay=wd)
                elif conf.optimizer == 'adam':
                    self.optimizer = optim.Adam(params, lr=conf.lr, weight_decay=wd)
                print(self.optimizer)
                #             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

                if conf.restore_suffix:
                    self.load_state(conf, conf.restore_suffix, from_save_folder=False, model_only=False)

                print('optimizers generated')
                self.board_loss_every = len(self.loader) // 100
                self.evaluate_every = len(self.loader) // 10
                self.save_every = len(self.loader) // 5

                self.board_loss_every = 20
                self.evaluate_every = len(self.loader)
                self.save_every = len(self.loader)
                if conf.data_mode == 'common':
                    import json
                    val_img_dir_map = json.loads(conf.val_img_dirs)
                    self.val_dataloaders = {}
                    for val_name in val_img_dir_map:
                        val_img_dir = val_img_dir_map[val_name]
                        print('not use pos', not conf.not_use_pos)
                        val_dataloader, common_val_issame = get_common_val_data(val_img_dir,
                                                                                conf.max_positive_cnt,
                                                                                conf.val_batch_size,
                                                                                conf.val_pin_memory,
                                                                                conf.num_workers,
                                                                                val_transforms=val_transforms,
                                                                                use_pos=not conf.not_use_pos,
                                                                                use_neg=not conf.not_use_neg)
                        self.val_dataloaders[val_name] = [val_dataloader, common_val_issame]
                else:
                    self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(
                        self.loader.dataset.root.parent)
            else:
                self.threshold = conf.threshold

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        import os
        print('save_path', save_path)
        print('save_path', str(save_path))
        os.makedirs(str(save_path), exist_ok=True)
        torch.save(
            self.model.state_dict(), save_path /
                                     ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step,
                                                                                   extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                                        ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy,
                                                                                     self.step,
                                                                                     extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                                             ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy,
                                                                                               self.step, extra)))

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        self.model.load_state_dict(torch.load(save_path / 'model_{}'.format(fixed_str)))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path / 'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor, pair_cnt=None):
        print(db_name, "step", self.step, "accuracy", accuracy, "best_threshold", best_threshold, "pair_cnt", pair_cnt)
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)

    #         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
    #         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
    #         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)

    def evaluate(self, conf, carray, issame, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def evaluate_by_dataloader(self, conf, val_dataloader, val_issame, nrof_folds=5):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(val_dataloader.dataset), conf.embedding_size])
        with torch.no_grad():
            is_grid = False
            for imgs in tqdm(iter(val_dataloader)):
                if not is_grid:
                    is_grid = True
                    grid = torchvision.utils.make_grid(imgs[:65])
                    grid = denormalize_image(grid)
                    self.writer.add_image('val_images', grid, self.step, dataformats='HWC')

                embeddings[idx:idx + len(imgs)] = self.model(imgs.to(conf.device)).cpu()
                idx += len(imgs)
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, val_issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss, batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            # Do the SGD step
            # Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.
        for e in range(epochs):
            if conf.train:
                print('epoch {} started'.format(e))
                if e == self.milestones[0]:
                    self.schedule_lr()
                if e == self.milestones[1]:
                    self.schedule_lr()
                if e == self.milestones[2]:
                    self.schedule_lr()
                # for imgs, labels in tqdm(iter(self.loader)):
                for imgs, labels in self.loader:
                    imgs = imgs.to(conf.device)
                    labels = labels.to(conf.device)
                    self.optimizer.zero_grad()
                    embeddings = self.model(imgs)
                    thetas = self.head(embeddings, labels)
                    loss = conf.ce_loss(thetas, labels)
                    loss.backward()
                    running_loss += loss.item()
                    self.optimizer.step()

                    if self.step % self.board_loss_every == 0 and self.step != 0:
                        loss_board = running_loss / self.board_loss_every
                        self.writer.add_scalar('train_loss', loss_board, self.step)
                        running_loss = 0.

                        grid = torchvision.utils.make_grid(imgs[:65])
                        grid = denormalize_image(grid)
                        self.writer.add_image('train_images', grid, self.step, dataformats='HWC')
                        print("epoch: {}, step: {}, loss: {}".format(e, self.step, loss_board))
                    # if self.step % self.evaluate_every == 0 and self.step != 0:
                    #     if conf.data_mode == 'common':
                    #         for val_name in self.val_dataloaders:
                    #             val_dataloader, val_issame = self.val_dataloaders[val_name]
                    #             accuracy, best_threshold, roc_curve_tensor = self.evaluate_by_dataloader(conf,
                    #                                                                                      val_dataloader,
                    #                                                                                      val_issame)
                    #             self.board_val(val_name, accuracy, best_threshold, roc_curve_tensor)
                    #     else:
                    #         accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30,
                    #                                                                    self.agedb_30_issame)
                    #         self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    #         accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    #         self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    #         accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp,
                    #                                                                    self.cfp_fp_issame)
                    #         self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    #     self.model.train()
                    # if self.step % self.save_every == 0 and self.step != 0:
                    #     self.save_state(conf, accuracy)

                    self.step += 1

            accuracies = []
            if conf.data_mode == 'common':
                for val_name in self.val_dataloaders:
                    val_dataloader, val_issame = self.val_dataloaders[val_name]
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate_by_dataloader(conf,
                                                                                             val_dataloader,
                                                                                             val_issame)
                    accuracies.append(accuracy)
                    self.board_val(val_name, accuracy, best_threshold, roc_curve_tensor, len(val_issame))
            else:
                accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30,
                                                                           self.agedb_30_issame)
                self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp,
                                                                           self.cfp_fp_issame)
                self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
            self.model.train()

            if not conf.train:
                break
            self.save_state(conf, sum(accuracies) / len(accuracies))

        if conf.train:
            self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def train_font(self, conf, epochs):
        self.model.train()
        running_loss = 0.
        for e in range(epochs):
            if conf.train:
                print('epoch {} started'.format(e))
                if e == self.milestones[0]:
                    self.schedule_lr()
                if e == self.milestones[1]:
                    self.schedule_lr()
                if e == self.milestones[2]:
                    self.schedule_lr()
                # for imgs, labels in tqdm(iter(self.loader)):
                for imgs, labels in self.loader:
                    imgs = imgs.to(conf.device)
                    labels = labels.to(conf.device)
                    self.optimizer.zero_grad()
                    embeddings = self.model(imgs)
                    thetas = self.head(embeddings, labels)
                    loss = conf.ce_loss(thetas, labels)
                    loss.backward()
                    running_loss += loss.item()
                    self.optimizer.step()

                    if self.step % self.board_loss_every == 0 and self.step != 0:
                        loss_board = running_loss / self.board_loss_every
                        self.writer.add_scalar('train_loss', loss_board, self.step)
                        running_loss = 0.

                        grid = torchvision.utils.make_grid(imgs[:65])
                        grid = denormalize_image(grid)
                        self.writer.add_image('train_images', grid, self.step, dataformats='HWC')
                        print("epoch: {}, step: {}, loss: {}".format(e, self.step, loss_board))
                    # if self.step % self.evaluate_every == 0 and self.step != 0:
                    #     if conf.data_mode == 'common':
                    #         for val_name in self.val_dataloaders:
                    #             val_dataloader, val_issame = self.val_dataloaders[val_name]
                    #             accuracy, best_threshold, roc_curve_tensor = self.evaluate_by_dataloader(conf,
                    #                                                                                      val_dataloader,
                    #                                                                                      val_issame)
                    #             self.board_val(val_name, accuracy, best_threshold, roc_curve_tensor)
                    #     else:
                    #         accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30,
                    #                                                                    self.agedb_30_issame)
                    #         self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    #         accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    #         self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    #         accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp,
                    #                                                                    self.cfp_fp_issame)
                    #         self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    #     self.model.train()
                    # if self.step % self.save_every == 0 and self.step != 0:
                    #     self.save_state(conf, accuracy)

                    self.step += 1

            accuracies = []
            if conf.data_mode == 'common':
                for val_name in self.val_dataloaders:
                    val_dataloader, val_issame = self.val_dataloaders[val_name]
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate_by_dataloader(conf,
                                                                                             val_dataloader,
                                                                                             val_issame)
                    accuracies.append(accuracy)
                    self.board_val(val_name, accuracy, best_threshold, roc_curve_tensor)
            else:
                accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30,
                                                                           self.agedb_30_issame)
                self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp,
                                                                           self.cfp_fp_issame)
                self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
            self.model.train()

            if not conf.train:
                break

            self.save_state(conf, sum(accuracies) / len(accuracies))

        if conf.train:
            self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)

    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)

        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum
