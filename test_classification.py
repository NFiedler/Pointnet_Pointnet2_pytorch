"""
Author: Benny
Date: Nov 2019
"""
from data_utils.KinectDataLoader import KinectDataLoader
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=2, type=int)
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--data_path', type=str, required=True, help='Data root')
    parser.add_argument('--tex_out', action='store_true', default=False, help='generate tex tables')
    parser.add_argument('--class_names', type=str, default='', help='class names separated by commas, no spaces')
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    predictions = torch.tensor([]).cpu()
    targets = torch.tensor([]).cpu()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]
        predictions = torch.cat((predictions, pred_choice.cpu()), 0)
        targets = torch.cat((targets, target.cpu()), 0)

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    labels = list(range(args.num_category))
    report = classification_report(targets, predictions, labels=labels)
    conf_matrix = confusion_matrix(targets, predictions, labels=labels)
    print(report)
    print(conf_matrix)
    if args.tex_out:
        class_names = args.class_names.split(',')
        report_dict = classification_report(targets, predictions, labels=labels, output_dict=True)
        acc_str = '''\\begin{table}[]
		\\caption{}
		\\label{tab:}
		\\centering
		\\begin{tabular}{l|r|r|r|r}
			         & Precision & Recall & F1-score & Support \\\\ 
			         <rows> \\hline\\hline
			Accuracy &           &        &     <acc> &     <supp> \\\\ \\hline
		\\end{tabular}
	\\end{table}'''
        rows = ''
        for i, class_name in enumerate(class_names):
            rows.append(f'\\hline\\\\\n{class_name} & {report_dict[str(i)]["precision"]} & {report_dict[str(i)]["recall"]} & {report_dict[str(i)]["f1-score"]} & {report_dict[str(i)]["support"]}')
        acc_str.replace('<rows>', rows)
        acc_str.replace('<acc>', report_dict['accuracy'])
        acc_str.replace('<supp>', report_dict['macro avg']['support'])
        with open(args.log_dir + '_eval.tex', 'w') as f:
            f.write(acc_str)

    with open(args.log_dir + '_eval.txt', 'w') as f:
        f.writelines([str(report), ' ', str(conf_matrix), ''])
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = args.data_path #'/homeL/5fiedler/data/t3_t4_t5_prep/'

    test_dataset = KinectDataLoader(root=data_path, split='test', include_normals=args.use_normals)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    class_names = args.class_names.split(',')
    num_class = test_dataset.get_num_classes()
    if args.tex_out:
        if len(class_names) != num_class:
            print('class names and class count mismatch!')
            print(num_class)
            print(class_names)
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
