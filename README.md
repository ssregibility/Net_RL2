train.py

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

parser.add_argument('--lambda1', default=0, type=float, help='lambda1 (for coeff loss) - currently disabled')

parser.add_argument('--lambda2', default=0.5, type=float, help='lambda2 (for basis loss)')

parser.add_argument('--rank', default=16, type=int, help='lambda2 (for basis loss)')

parser.add_argument('--dataset', default="CIFAR100", help='CIFAR10, CIFAR100')

parser.add_argument('--batch_size', default=256, type=int, help='batch_size')

parser.add_argument('--model', default="ResNet34", help='ResNet152, ResNet101, ResNet50, ResNet34, ResNet18, ResNet34_Basis, ResNet18_Basis')

parser.add_argument('--visible_device', default="0", help='CUDA_VISIBLE_DEVICES')

