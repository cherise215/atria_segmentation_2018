import sys
import os
from optparse import OptionParser
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch


from networks.utils import get_scheduler
from networks.multi_task_unet import MT_Net,cal_cls_acc
from networks.myloss import  cross_entropy_2D
from data_io.atria_dataset import AtriaDataset
from utils import runningScore
from networks.init_weight import init_weights
from tensorboardX import  SummaryWriter

IMAGE_SIZE_LIST = [256, 384, 480, 512, 576, 640]
SPP_GRID = [8, 4, 1]

def train_net(sequence,orientation,root_dir,model_name,net,n_classes,csv_path, epochs=5, batch_size=4, lr=0.1,
              cp=True,gpu=False,if_clahe=False, if_gamma_correction=False,if_mip=False,dir_checkpoints='models/'):

    global IMAGE_SIZE_LIST
    image_size_list=IMAGE_SIZE_LIST
    # set up paramerter
    optimizer = optim.SGD(net.parameters(),
                          lr=lr, momentum=0.99, weight_decay=0.0005)

    best_iou = -100.0
    start_epoch=0
    if not os.path.exists(dir_checkpoints):
        os.mkdir(dir_checkpoints)

    print('''
           Starting training:
               Model_name:{}
               Epochs: {}
               Batch size: {}
               Learning rate: {}
           '''.format(model_name,epochs, batch_size, lr))

    running_metrics = runningScore(n_classes)
    scheduler = get_scheduler(optimizer, lr_policy='step', lr_decay_iters=50)
    ## if disk exists model
    resume_path = os.path.join(dir_checkpoints,options.model_name+'.pkl')
    if resume_path is not None:
        if os.path.isfile(resume_path):
            print("Loading model and optimizer from checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            net.module.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            start_epoch = int(checkpoint['epoch'])
        else:
            print("No checkpoint found at '{}'".format(resume_path))
    else:
        init_weights(net, init_type='kaiming')

    temp_batch_size=batch_size
    for epoch in range(start_epoch,epochs):
        scheduler.step(epoch)
        total=0.
        acc=0.
        for size in image_size_list:
            if orientation==0:
                w=h=size
                if h>=480 or w>=480: ##for memomry concern
                    batch_size=int(temp_batch_size/2)
                    if h >= 600 or w >= 600:
                        batch_size = int(temp_batch_size / 4)
                else:
                    batch_size=int(temp_batch_size)
            else:
                h=96
                w=size
                batch_size=temp_batch_size
            if batch_size==0:
                batch_size=1
            # Setup Dataloader
            train_dataset = AtriaDataset(root_dir, if_subsequent=sequence,sequence_length=options.sequence_length,split='train',extra_label_csv_path=csv_path,extra_label=True,augmentation=True,input_h=h,input_w=w,preload_data=False,if_clahe=if_clahe,if_gamma_correction=if_gamma_correction,if_mip=if_mip,orientation=orientation)
            train_loader = DataLoader(dataset=train_dataset,num_workers=16, batch_size=batch_size, shuffle=True)
            test_dataset = AtriaDataset(root_dir, if_subsequent=sequence,split='validate',sequence_length=options.sequence_length,extra_label_csv_path=csv_path,extra_label=True,augmentation=True,input_h=h,input_w=w, preload_data=True,if_clahe=if_clahe,if_gamma_correction=if_gamma_correction,if_mip=if_mip,orientation=orientation)
            test_loader = DataLoader(dataset=test_dataset, num_workers=16, batch_size=batch_size, shuffle=True)

            print('''
                             Starting training:
                                 model_name:{}
                                 lr: {}
                                 Image size: {}
                                 Training size: {}
                                 Validation size: {}
                                 Checkpoints: {}
                                 CUDA: {}
                             '''.format(model_name, str(scheduler.get_lr()),size, train_dataset.get_size(),
                                        test_dataset.get_size(), str(cp), str(gpu)))
            net.train()
            print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
            for epoch_iter,data in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
                images = data['input']
                labels = data['target']
                gt_pa_label =  data['post_ablation']
                if gpu:
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                    gt_pa_label = Variable(gt_pa_label.cuda())
                else:
                    images = Variable(images)
                    labels = Variable(labels)
                    gt_pa_label = Variable(gt_pa_label)

                optimizer.zero_grad()
                if isinstance(net, torch.nn.DataParallel):
                    name=net.module.get_net_name()
                else:
                    name=net.get_net_name()
                print ('network',name)

                outputs=net(images)
                loss = cross_entropy_2D(input=outputs[0], target=labels)
                logits=F.sigmoid(outputs[1])

                classifier_fn = nn.CrossEntropyLoss()
                cs_loss = classifier_fn(input=logits, target=gt_pa_label)
                loss+=cs_loss

                loss.backward()
                optimizer.step()

                if (epoch_iter + 1) % 20 == 0:
                    print("Epoch [%d/%d] Loss: %.4f" % (epoch+ 1, epochs, loss.item()))
                    writer.add_scalar(options.model_name + '/loss',  loss.item(), epoch+1)


            net.eval()
            for i_val,data in tqdm(enumerate(test_loader)):
                images_val = data['input']
                labels_val = data['target']
                gt_pa_label = data['post_ablation']
                if gpu:
                    images_val = Variable(images_val.cuda())
                    labels_val = Variable(labels_val.cuda())
                else:
                    images_val = Variable(images_val)
                    labels_val = Variable(labels_val)

                with torch.no_grad():
                    outputs = net(images_val)
                sum, count = cal_cls_acc(outputs[1], gt_pa_label)
                total += count
                acc += sum

                pred = outputs[0].data.max(1)[1].cpu().numpy()
                gt = labels_val.data.cpu().numpy()
                ## segmentation result evaluate
                running_metrics.update(gt, pred)

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        writer.add_scalars(options.model_name + '/scalar_group',score, epoch+1)
        print('classification acc:',100 * acc / (1.0*total))
        writer.add_scalar(options.model_name + '/classification',100 * acc / (1.0*total), epoch+1)


        acc=0.
        total=0.
        running_metrics.reset()
        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch + 1,
                     'model_state': net.module.state_dict(),
                     'optimizer_state': optimizer.state_dict(),
                     'if_mip':if_mip,
                     'if_gamma':if_gamma_correction,
                     'if_clahe':if_clahe,
                     'orientation':options.orientation,
                     'sequence_length':options.sequence_length,
                     'upsample_type':options.upsample_type
                     }
            torch.save(state, os.path.join(dir_checkpoints,model_name+".pkl"))

if __name__ == '__main__':
        parser = OptionParser()
        parser.add_option('--root_dir', dest='root_dir', default='/vol/medic01/users/cc215/data/AtriaSeg_2018_training/dataset', type=str,
                          help='dataset_path')
        parser.add_option('-e', '--epochs', dest='epochs', default=2000, type='int',
                          help='number of epochs')
        parser.add_option( '--n_classes', dest='n_classes', default=2, type='int',
                          help='number of class')
        parser.add_option('-b', '--batch-size', dest='batchsize', default=8,
                          type='int', help='batch size')
        parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                          type='float', help='learning rate')
        parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                          default=True, help='use cuda')
        parser.add_option( '--num_gpus', type='int', dest='n_gpu',
                          default=1, help='the number of gpus used for training')
        parser.add_option('-c', '--load', dest='load',
                          default=False, help='load file model')
        parser.add_option('-r','--resume_path',type=str, dest='resume',default=None,
                            help='Path to previous saved model to restart from')
        parser.add_option('-a', '--adaenhance',  action='store_true',dest='enhance',default=False, help='use enhance to preprocess image')
        parser.add_option( '--gamma',  action='store_true',dest='gamma',default=False, help='use gamma enhance to preprocess image')
        parser.add_option('--mip', action='store_true', dest='mip', default=False,
                          help='use mip as extra channel')
        parser.add_option('--name', type=str, dest='model_name',
                          help='save model name')
        parser.add_option('--orientation', type=str, dest='orientation',
                          help='training data orientation;axial,coronal,sagittal',default='axial')
        parser.add_option('--mt', action='store_true', dest='MT',
                          help='training data with classfication task', default=False)
        parser.add_option('--sequence', action='store_true', dest='sequence',
                          help='use pre+post slice as information', default=False)
        parser.add_option('--sequence_length', type=int, dest='sequence_length',
                          help='use how many sequence number', default=1)
        parser.add_option('--upsample_type', type=str, dest='upsample_type',
                          help='use  which method for upsampling, bilinear, bilinear_additive or deconv', default='bilinear')
        parser.add_option('--csv_path',type=str,dest='csv_path',help='a csv file path recording pre or post ablation',default='./data/pre post ablation recrods.csv')
        (options, args) = parser.parse_args()
        orientation_dict={'axial':0,'coronal':1,'sagittal':2}
        orientation=orientation_dict[options.orientation]
        print ('orientation: {}'.format(str(orientation)))
        writer = SummaryWriter(comment='training:'+options.model_name)

        root_dir=options.root_dir
        net = MT_Net(n_channels=options.sequence_length, spp_grid=SPP_GRID, n_classes=2, n_labels=2,
                   if_dropout=True, upsample_type=options.upsample_type)

        if options.load:
            net.load_state_dict(torch.load(options.load))
            print('Model loaded from {}'.format(options.load))

        if options.gpu:
            gpu_id_list=[i for i in range(options.n_gpu)]
            net = torch.nn.DataParallel(net, device_ids=gpu_id_list)
            net.cuda()
            cudnn.benchmark = True
        if options.gamma:
            print ('gamma correction:'+str(options.gamma))
        if options.enhance:
            print ('enhance :'+str(options.enhance))


        try:
            train_net(
                   sequence=options.sequence,
                   orientation=orientation,
                   root_dir=root_dir,
                   model_name=options.model_name,
                   net=net,
                   csv_path=options.csv_path,
                   epochs=options.epochs,
                   batch_size=options.batchsize,
                   lr=options.lr,
                   n_classes=options.n_classes, gpu=options.gpu,if_clahe=options.enhance, if_gamma_correction=options.gamma,if_mip=options.mip)
            writer.export_scalars_to_json("./" + options.model_name + "_train.json")
            writer.close()
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
