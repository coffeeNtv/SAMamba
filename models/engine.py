import os
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored

import pandas as pd
import torch.optim
import torch.nn.parallel


class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold

        # tensorboard
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()

        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.validate(val_loader, model, criterion)
            return

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer)
            # evaluate on validation set
            c_index, risk_csv = self.validate(val_loader, model, criterion)
            # remember best c-index and save checkpoint
            is_best = c_index > self.best_score
            if is_best:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score})
                risks = risk_csv
            print(' *** best c-index={:.4f} at epoch {}'.format(self.best_score, self.best_epoch))
            if scheduler is not None:
                if self.args.scheduler == 'plateau':
                    scheduler.step(self.metric)
                else:
                    scheduler.step()

        return self.best_score, self.best_epoch, risks

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        train_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Train Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI, omic_list, label, event_time, c, case_id) in enumerate(dataloader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_omics = []
                for item in omic_list:
                    data_omics.append(item.type(torch.FloatTensor).cuda())
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            hazards, S = model(x_path=data_WSI, x_omics=data_omics)
            if torch.isnan(S).any():
                continue
                
            loss = criterion(hazards=hazards, S=S, Y=label, c=c)

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            
        # calculate loss and error for epoch
        train_loss /= len(dataloader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(train_loss, c_index))

        if self.writer:
            self.writer.add_scalar('train/loss', train_loss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)

    def validate(self, data_loader, model, criterion):
        model.eval()
        val_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        case_id_list = []
        risk_list = []

        dataloader = tqdm(data_loader, desc='Test Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI, omic_list, label, event_time, c, case_id) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_omics = []
                for item in omic_list:
                    data_omics.append(item.type(torch.FloatTensor).cuda())
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            with torch.no_grad():
                hazards, S = model(x_path=data_WSI, x_omics=data_omics)  # return hazards, S, Y_hat, A_raw, results_dict
                if torch.isnan(S).any():
                    continue
                    
            loss = criterion(hazards=hazards, S=S, Y=label, c=c)
            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time
            val_loss += loss.item()
            case_id_list.extend(case_id)
            risk_list.extend(risk.tolist())
        assert len(case_id_list) == len(risk_list)
        risk_csv = pd.DataFrame({'case_id':case_id_list,'risk':risk_list})
 

        val_loss /= len(dataloader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(val_loss, c_index))
        print('loss: {:.4f}, c_index: {:.4f}'.format(val_loss, c_index))
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, self.epoch)
            self.writer.add_scalar('val/c-index', c_index, self.epoch)
        return c_index, risk_csv

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir, 'fold_' + str(self.fold), 'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
