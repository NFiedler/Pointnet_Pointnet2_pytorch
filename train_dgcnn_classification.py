import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.KinectDataLoader import KinectDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

class Trainer:

    def __init__(self):

        self.args = self.parse_args()

        '''HYPER PARAMETER'''
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        self.out = True

        '''CREATE DIR'''
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        self.exp_dir = Path('./log/')
        self.exp_dir.mkdir(exist_ok=True)
        self.exp_dir = self.exp_dir.joinpath('classification')
        self.exp_dir.mkdir(exist_ok=True)
        if self.args.log_dir is None:
            exp_dir = self.exp_dir.joinpath(timestr)
        else:
            exp_dir = self.exp_dir.joinpath(self.args.log_dir)
        exp_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = exp_dir.joinpath('checkpoints/')
        self.checkpoints_dir.mkdir(exist_ok=True)
        log_dir = exp_dir.joinpath('logs/')
        log_dir.mkdir(exist_ok=True)

        '''LOG'''
        self.logger = logging.getLogger("Model")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, self. args.model))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.log_string('PARAMETER ...')
        self.log_string(self.args)


    def parse_args(self):
        '''PARAMETERS'''
        parser = argparse.ArgumentParser('training')
        parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
        parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
        parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
        parser.add_argument('--model', default='dgcnn_cls', help='model name [default: dgcnn_cls]')
        parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
        parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
        parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
        parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
        parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
        parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
        parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
        parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
        parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
        parser.add_argument('--data_path', type=str, required=True, help='Data root')
        return parser.parse_args()


    def inplace_relu(self, m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace = True

    def load_data(self, batch_size=None, num_points=None):
        self.log_string('Load dataset ...')
        if not batch_size:
            batch_size = self.args.batch_size
        if not num_points:
            num_points = self.args.num_points
        data_path = self.args.data_path
        self.train_dataset = KinectDataLoader(root=data_path, split='train', include_normals=self.args.use_normals, num_points=num_points, center_pointclouds=True, random_scaling=True)
        self.test_dataset = KinectDataLoader(root=data_path, split='test', include_normals=self.args.use_normals, num_points=num_points, center_pointclouds=True, random_scaling=True)
        self.trainDataLoader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
        self.testDataLoader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    def test(self, model, num_class=40):
        mean_correct = []
        class_acc = np.zeros((num_class, 3))
        classifier = model.eval()

        # for j, (points, target) in tqdm(enumerate(self.testDataLoader), total=len(self.testDataLoader)):
        for j, (points, target) in enumerate(self.testDataLoader):

            if not self.args.use_cpu:
                points, target = points.cuda(), target.cuda()

            points = points.transpose(2, 1)
            pred, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                class_acc[cat, 1] += 1

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = np.mean(class_acc[:, 2])
        instance_acc = np.mean(mean_correct)

        return instance_acc, class_acc

    def log_string(self, str):
        if self.out:
            self.logger.info(str)
            print(str)

    def train(self, batch_size, num_points, learning_rate, k, dropout, emb_dims, modelname, use_normals, decay_rate, sched_step_size, sched_gamma, trial=None):

        self.load_data(batch_size=batch_size, num_points=num_points)
        num_class = self.train_dataset.get_num_classes()
        if self.out:
            self.log_string('Classes:')
            self.log_string(num_class)

        '''MODEL LOADING'''
        model = importlib.import_module(modelname)
        shutil.copy('./models/%s.py' % modelname, str(self.exp_dir))
        shutil.copy('models/pointnet2_utils.py', str(self.exp_dir))
        shutil.copy('./train_dgcnn_classification.py', str(self.exp_dir))

        classifier = model.get_model(num_class=num_class, k=k, dropout=dropout, emb_dims=emb_dims, normal_channel=use_normals)
        criterion = model.get_loss()
        classifier.apply(self.inplace_relu)

        if not self.args.use_cpu:
            classifier = classifier.cuda()
            criterion = criterion.cuda()

        try:
            checkpoint = torch.load(str(self.exp_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            if self.out:
                self.log_string('Use pretrain model')
        except:
            if self.out:
                self.log_string('No existing model, starting training from scratch...')
            start_epoch = 0

        if self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=decay_rate
            )
        else:
            optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step_size, gamma=sched_gamma)
        global_epoch = 0
        global_step = 0
        best_instance_acc = 0.0
        best_class_acc = 0.0

        '''TRANING'''
        if self.out:
            self.logger.info('Start training...')
        for epoch in range(start_epoch, self.args.epoch):
            if self.out:
                self.log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, self.args.epoch))
            mean_correct = []
            classifier = classifier.train()

            # for batch_id, (points, target) in tqdm(enumerate(self.trainDataLoader, 0), total=len(self.trainDataLoader), smoothing=0.9):
            for batch_id, (points, target) in enumerate(self.trainDataLoader, 0):
                optimizer.zero_grad()
                points = points.data.numpy()
                points = provider.random_point_dropout(points)
                #points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)
                #target = torch.Tensor(target)
                points = points.transpose(2, 1)

                if not self.args.use_cpu:
                    # points = points.cuda()
                    # target = target.cuda()
                    points, target = points.cuda(), target.cuda()

                pred, trans_feat = classifier(points)
                loss = criterion(pred, target.long(), trans_feat)
                pred_choice = pred.data.max(1)[1]

                correct = pred_choice.eq(target.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))
                loss.backward()
                optimizer.step()
                global_step += 1

            train_instance_acc = np.mean(mean_correct)
            if self.out:
                self.log_string('Train Instance Accuracy: %f' % train_instance_acc)

            with torch.no_grad():
                instance_acc, class_acc = self.test(classifier.eval(), num_class=num_class)

                if (instance_acc >= best_instance_acc):
                    best_instance_acc = instance_acc
                    best_epoch = epoch + 1

                if (class_acc >= best_class_acc):
                    best_class_acc = class_acc
                if self.out:
                    self.log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
                    self.log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

                if (instance_acc >= best_instance_acc):
                    savepath = str(self.checkpoints_dir) + '/best_model.pth'
                    if self.out:
                            self.logger.info('Save model...')
                            self.log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': best_epoch,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                global_epoch += 1
            scheduler.step()
        if self.out:
            self.log_string('End of training...')


if __name__ == '__main__':
    t = Trainer()
    t.train()
