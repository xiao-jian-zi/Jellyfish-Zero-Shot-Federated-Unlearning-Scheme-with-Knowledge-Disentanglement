import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
from torch.nn import functional as F
from torch.autograd import Variable
from tqdm import tqdm
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(device)
model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./cifar', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)

best_prec1 = 0
args = parser.parse_args()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

full_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    normalize,
]), download=True)

split_indices = list(range(0,5000))
train_subset = torch.utils.data.Subset(full_dataset, split_indices)
train_subset = list(train_subset)
forget_class = [1]

forget_set = []
remain_set = []
for i in range(len(train_subset)):
    train_subset[i] = list(train_subset[i])
    if train_subset[i][1] in forget_class :
        forget_set.append(train_subset[i])
    else:
        remain_set.append(train_subset[i])
        
print('forget_set:',len(forget_set))


train_loader = torch.utils.data.DataLoader(full_dataset,
                                           batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True
                                           )
train_loader1 = torch.utils.data.DataLoader(train_subset,
                                           batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True
                                           )

Forget_loader = torch.utils.data.DataLoader(forget_set,
                                           batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True
                                           )
Remain_loader = torch.utils.data.DataLoader(remain_set,
                                           batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True
                                           )

val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    normalize,
]))

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=128, shuffle=False,
    num_workers=args.workers, pin_memory=True)


batch_sizes = [len(batch) for _,(batch,_) in enumerate(Forget_loader)]
Noise_dataset = [  torch.randn(batch_size, full_dataset[0][0].size(0), full_dataset[0][0].size(1), full_dataset[0][0].size(2)).to(device).requires_grad_(True)  for batch_size in  batch_sizes]
Noise_optimizer = torch.optim.Adam(Noise_dataset, lr=0.005)

def train_noise(noise_epochs):
    
    classifier.eval()
    test(val_loader,classifier)
    print('start decoupling')
    for epoch in tqdm(range(noise_epochs),desc='training',unit='epoch'):
        for i,data in enumerate(Forget_loader, 0):
            imgs, labels = data
            imgs, labels = Variable(imgs), Variable(labels)
            imgs, labels = imgs.to(device), labels.to(device)
            Noise_optimizer.zero_grad()
            noise_pred, _ = classifier(Noise_dataset[i])
            imgs_pred , _ = classifier(imgs)
            imgs_pred  = F.softmax(imgs_pred  , dim=1)
            loss =  (
                     + criterion(noise_pred, labels.long()) 
                    )
            # print(loss.item())
            loss.backward()
            Noise_optimizer.step()
            

classifier =         torch.nn.DataParallel(resnet.__dict__[args.arch](), device_ids=[5])
classifier.load_state_dict(torch.load('./saves/resnet32-d509ac18.th')['state_dict'])

teacher_classifier = torch.nn.DataParallel(resnet.__dict__[args.arch](), device_ids=[5])
teacher_classifier.load_state_dict(torch.load('./saves/resnet32-d509ac18.th')['state_dict'])

teacher_classifier.eval()
teacher_params = {k: v for k, v in teacher_classifier.named_parameters()}

random_classifier1 = torch.nn.DataParallel(resnet.__dict__[args.arch](), device_ids=[5])
random_classifier2 = torch.nn.DataParallel(resnet.__dict__[args.arch](), device_ids=[5])
random_classifier3 = torch.nn.DataParallel(resnet.__dict__[args.arch](), device_ids=[5])

criterion = nn.CrossEntropyLoss().to(device)
if args.half:
        classifier.half()
        criterion.half()
optimizer =          torch.optim.SGD(classifier.parameters(), lr=0.3, momentum=0.9, weight_decay=0)
optimizer_decouple = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=0)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[100, 150], last_epoch=args.start_epoch - 1)

def train(unlearn_epochs , disentangle_epochs):
    batch_size = 128
    print('-----------start decoupling-----------')   
    for epoch in tqdm(range(disentangle_epochs),desc='training',unit='epoch'):
        decouple_losses = []
        optimizer_decouple.zero_grad()
        for i,data in enumerate(Forget_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            G_real_labels = torch.full((labels.shape[0],), 1., device=device)
            G_real_outputs, lastconv = classifier(Noise_dataset[i].detach())
            decouple_losses.append(calculate_decouple_loss(lastconv,Thr = 0.9) *0.1)
        decouple_loss = sum(decouple_losses)/len(decouple_losses)
        # print(decouple_loss.item())
        decouple_loss.backward()
        optimizer_decouple.step()
        test(val_loader,classifier)
    
    
    print('-----------start unlearning-----------')
    for epoch in tqdm(range(unlearn_epochs),desc='training',unit='epoch'):
        classifier.train()
        random_classifier1.eval()
        random_classifier2.eval()
        random_classifier3.eval()
        losses = []
        for i,data in enumerate(Forget_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            G_real_labels = torch.full((labels.shape[0],), 1., device=device)
            G_real_outputs, _ = classifier(Noise_dataset[i].detach())
            # G_real_outputs = F.softmax(G_real_outputs , dim=1)
            with torch.no_grad():
                teacher_outputs ,_    = teacher_classifier(Noise_dataset[i].detach())
                G_random_outputs1, _ = random_classifier1(Noise_dataset[i].detach())
                G_random_outputs2, _ = random_classifier2(Noise_dataset[i].detach())
                G_random_outputs3, _ = random_classifier3(Noise_dataset[i].detach())
                
            probs = F.softmax(teacher_outputs, dim=1)
            for a in range(probs.size(0)):
                sample_probs = probs[a]
                sorted_indices = sample_probs.sort(dim=0, descending=True)[1]
                G_real_labels[a] = sorted_indices[1].item()
            
            G_realinputs_loss =  (
                - criterion( G_real_outputs, labels.long()) * 2
                + criterion( teacher_outputs, G_real_labels.long())
                + distillation_loss(G_real_outputs, G_random_outputs1, T = 2)/3
                + distillation_loss(G_real_outputs, G_random_outputs2, T = 2)/3
                + distillation_loss(G_real_outputs, G_random_outputs3, T = 2)/3
            )
            
            losses.append(G_realinputs_loss / batch_size)
        avg_loss = sum(losses) / len(losses)
        print(avg_loss.item())
        
        # save realinputs grad
        G_realinputs_grad = torch.autograd.grad(  avg_loss, classifier.parameters(), retain_graph=True,create_graph=True)
        masks = threshold_mask(G_realinputs_grad, retain_percentage=75)
        current_params = {k: v for k, v in classifier.named_parameters()}
        params_diff_loss = sum((current_params[k] - teacher_params[k]).pow(2).sum() for k in current_params) 
        print(params_diff_loss, params_diff_loss.item())
        update = torch.autograd.grad( params_diff_loss, classifier.parameters(), retain_graph=True,create_graph=True)
        update_masked = apply_mask_to_grads(update, masks)
        
        Gradient = gradient_harmonization(g_g = update_masked, g_f = G_realinputs_grad)
        
        optimizer.zero_grad()
        for param, computed_grad in zip(classifier.parameters(), Gradient):
            param.grad = computed_grad
        optimizer.step()

        test(train_loader1,classifier)
        test(val_loader,classifier)

    

def threshold_mask(grads, retain_percentage=20):
    """
    Create a mask using a threshold to retain the top percentage of parameters.

    Parameters:
    grads (list of tensors): List of gradient tensors.
    retain_percentage (float): Percentage of parameters to retain (default: 20).

    Returns:
    list of tensors: List of mask tensors.
    """
    # Flatten all gradient tensors
    flat_grads = [grad.view(-1) for grad in grads] 
    all_grads = torch.cat(flat_grads)
    # Calculate the threshold to retain the specified percentage of parameters
    num_params_to_keep = int((retain_percentage / 100) * len(all_grads))
    # print(num_params_to_keep)
    # Ensure at least one parameter is retained
    if num_params_to_keep == 0:
        num_params_to_keep = 1
    # Calculate the threshold value
    threshold = torch.kthvalue(torch.abs(all_grads), num_params_to_keep - 1).values
    # print(threshold)
    # Create masks for each gradient tensor
    masks = []
    for grad in grads:
        # Ensure the mask has the same data type as the original gradient
        grad_dtype = grad.dtype
        grad_device = grad.device
        mask = (torch.abs(grad) < torch.tensor(threshold, dtype=grad_dtype, device=grad_device)).to(torch.float32)
        masks.append(mask)
    return masks

def apply_mask_to_grads(grads, masks):
    """
    Apply masks to the original gradients.

    Parameters:
    grads (list of tensors): List of gradient tensors.
    masks (list of tensors): Corresponding list of mask tensors.
    """
    masked_grads = [grad * mask for grad, mask in zip(grads, masks)]
    return masked_grads
def gradient_harmonization(g_g, g_f):
    """
    Gradient Harmonization function to harmonize the gradients of remembering (g_g) and forgetting (g_f).
    
    Parameters:
    g_g (list of torch.Tensor): The gradients for remembering (update).
    g_f (list of torch.Tensor): The gradients for forgetting (G_realinputs_grad_neg).
    
    Returns:
    list of torch.Tensor: The harmonized gradients.
    """
    # # Check if the gradients are lists
    # if not isinstance(g_g, list) or not isinstance(g_f, list):
    #     raise ValueError("Both g_g and g_f must be lists of torch.Tensor.")
    # Check if the lists have the same length
    if len(g_g) != len(g_f):
        raise ValueError("g_g and g_f must have the same number of elements.")
    
    # Initialize the harmonized gradients list
    harmonized_gradients = []
    
    # Iterate over the gradients
    for grad_g, grad_f in zip(g_g, g_f):
        # Calculate the cosine similarity between the gradients
        cos_sim = torch.sum(grad_g * grad_f) / (torch.norm(grad_g) * torch.norm(grad_f))
         
        # If the cosine similarity is less than zero, harmonize the gradients
        if cos_sim < 0:
            # Project grad_f onto the subspace orthogonal to grad_g
            proj_grad_f = (grad_f * grad_g) / (grad_g ** 2).sum() * grad_g
            g_prime_f = grad_f - proj_grad_f
            harmonized_gradients.append(grad_g + g_prime_f)
        else:
            # If not conflicting, just add the gradients
            harmonized_gradients.append(grad_g + grad_f)
    
    return harmonized_gradients
def calculate_decouple_loss(last_conv_output, Thr):
    """
    Calculate the decouple loss based on the last convolutional layer's output.

    Parameters:
    last_conv_output (tensor): Output of the last convolutional layer.
    Thr (float): Threshold for retaining channels.

    Returns:
    tensor: Decouple loss.
    """
    # Get the dimensions of the feature map
    batchsize, height, width = last_conv_output.size(0), last_conv_output.size(2), last_conv_output.size(3)
    # Calculate the L1 norm of each channel's feature map and divide by the number of elements to get the average norm
    norms = torch.norm(last_conv_output, p=1, dim=(0, 2, 3)) / (batchsize * height * width)
    # Select channels based on the average norm size
    num_channels = last_conv_output.size(1)
    num_channels_to_keep = int(num_channels * Thr)  # Calculate the number of channels to retain based on the threshold
    sorted_norms, _ = torch.sort(norms, descending=True)  # Sort all channel norms in descending order
    threshold = sorted_norms[num_channels_to_keep]  # Get the threshold value
    # Sum the average norms of channels outside the retention ratio
    mask = sorted_norms < threshold  # Create a mask where channels with average norms less than the threshold are False
    masked_norms = sorted_norms * mask.float()  # Apply the mask
    loss = torch.sum(masked_norms)  # Sum the masked norms
    return loss

def cosine_similarity_loss(grad1, grad2):
    """
    Calculate the cosine similarity loss between two gradients.

    Parameters:
    grad1 (tensor): First gradient tensor.
    grad2 (tensor): Second gradient tensor.

    Returns:
    tensor: Cosine similarity loss.
    """
    # First, normalize the gradients using L2 norm
    grad1_norm = F.normalize(grad1, p=2, dim=0)
    grad2_norm = F.normalize(grad2, p=2, dim=0)
    # Calculate the dot product of the normalized gradients
    dot_product = (grad1_norm * grad2_norm).sum(dim=0).mean()
    # Cosine similarity = dot product / (norm1 * norm2)
    norm1 = grad1_norm.norm(p=2, dim=0).mean()  # L2 norm of grad1
    norm2 = grad2_norm.norm(p=2, dim=0).mean()  # L2 norm of grad2
    cosine_similarity = dot_product / (norm1 * norm2 + 1e-8)  # Add a small constant to avoid division by zero
    return cosine_similarity.requires_grad_()

def L2loss(list1 , list2):
    loss = 0
    for grad1,grad2 in zip(list1 , list2):
        loss += (grad1 - grad2).sum().abs()
    return (0.5 * loss**2).requires_grad_()
        
def compare_updata(g_fakeinputs_grad,g_realinputs_grad):
    cosine_similarities = []
    for (param1,param2) in zip(g_fakeinputs_grad,g_realinputs_grad):
        # if param1.grad is not None and param2.grad is not None:
            similarity = cosine_similarity_loss(param1, param2)
            cosine_similarities.append(similarity)
    if cosine_similarities:
        return sum(cosine_similarities) / len(cosine_similarities)
    else:
        return None    
            
            
def distillation_loss(outputs,teacher_outputs,T):
    # hard_loss = nn.NLLLoss().to(device)
    soft_loss=nn.KLDivLoss(reduction="batchmean")
    eps=1e-6
    ditillation_loss=(soft_loss((F.softmax(outputs/T,dim=1)+eps).log(),F.softmax(teacher_outputs/T,dim=1))+
                      soft_loss((F.softmax(teacher_outputs/T,dim=1)+eps).log(),F.softmax(outputs/T,dim=1)))/2
    return ditillation_loss

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            if args.half:
                input_var = input_var.half()

            # compute output
            output,_ = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


def test(testloader,net):
    correct = 0
    total = 0
    
    for data in testloader:
        images, labels = data
        labels = labels.to(device)
        outputs,_ = net(Variable(images).to(device))
        # print(F.softmax(outputs,dim=1))
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.to(device)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    # print(correct,total)
    # print('Accuracy of the network on the 10000 test images: %.2f %%' % (100*correct/total))
    print('%.2f %%'%(100*correct/total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        labels = labels.to(device)
        outputs,_ = net(Variable(images).to(device))
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.to(device)
        c = (predicted == labels).squeeze()
        # print(predicted)
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    Df_correct = 0; Df_total = 0
    for i in forget_class:
        Df_correct += class_correct[i]
        Df_total += class_total[i]
    print('Df_acc: ', (Df_correct/Df_total).item()*100)
    print('########################################################')
    # for i in range(10):
    #     if class_total[i] != 0:
    #         prin_tresult = 100 * class_correct[i] / class_total[i]
    #     else:
    #         prin_tresult = 0
    #     print(prin_tresult.item()/100)

if __name__ == '__main__':
    train_noise(noise_epochs = 200)
    torch.save(Noise_dataset, './saves/cifarnoise.pth')
    Noise_dataset = torch.load('./saves/cifarnoise.pth')
    train(unlearn_epochs= 100, disentangle_epochs = 5)