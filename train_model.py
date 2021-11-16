import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
import os
import time
import itertools
from sklearn.metrics import f1_score, accuracy_score
import onnx
import onnx.numpy_helper as numpy_helper

start_running_time = time.time()
start_train_time = time.time()

# 0. GPU 인식 여부 확인
print("torch version:", torch.__version__)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("cpu")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# 사용방법: 명령어.to(device) → GPU환경, CPU환경에 맞춰서 동작

# target = "periocular"
target = "full_face"

# 1. 설정
class Config:
    training_dir = "./dataset/db_" + target + "/train"
    train_batch_size = 64
    train_number_epochs = 300
    testing_dir = "./dataset/db_" + target + "/test"
    test_balance = 140

# 2. Dataset 설정
class SiameseNetworkDataset(Dataset):
    # 0: Geunine Pair, 1: Imposter Pair
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # print (img0_tuple[1]) : folder name
        # we need to make sure approx 50% of images are in the same class
        # print ("img0 ",img0_tuple[0], img0_tuple[1])
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)

                if (img0_tuple[1] == img1_tuple[1]) and (img0_tuple[0] != img1_tuple[0]):
                    # print ("img1 ", img1_tuple[0], "genuine", img1_tuple[1])
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)

                if img0_tuple[1] != img1_tuple[1]:
                    # print ("img1 ", img1_tuple[0],"imposter",img1_tuple[1])
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        img0_folder = os.path.dirname(img0_tuple[0])[-4:]
        img1_folder = os.path.dirname(img1_tuple[0])[-4:]

        return img0, img1, torch.from_numpy(
            np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)), img0_folder, img1_folder, img0_tuple[0], \
               img1_tuple[0]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# 3. Periocular 네트워크 모델 정의
class SiameseNetwork(nn.Module):
    # 입력: 이미지 2장 → 출력: 길이 100의 vector 2개
    # torch.Size([batch_size, 1, 105, 105]) → torch.Size([batch_size, 100])
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.ConvolutionalLayer = nn.Sequential(
            # Conv2d: 입력채널 수, 출력채널 수, 필터크기, 패딩
            nn.Conv2d(1, 64, kernel_size=10, padding=0),  # 105*105*1 → 96*96*64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.MaxPool2d((2, 2), stride=(2, 2)),  # 96*96*64 → 48*48*64

            nn.Conv2d(64, 128, kernel_size=7, padding=0),  # 48*48*64 → 42*42*128
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.MaxPool2d((2, 2), stride=(2, 2)),  # 42*42*128 → 21*21*128

            nn.Conv2d(128, 128, kernel_size=4, padding=0),  # 21*21*128 → 18*18*128
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.MaxPool2d((2, 2), stride=(2, 2)),  # 18*18*128 → 9*9*128

            nn.Conv2d(128, 256, kernel_size=4, padding=0),  # 9*9*128 → 6*6*256 (9*9*256)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        self.FullyConnectedLayer = nn.Sequential(
            nn.Linear(6 * 6 * 256, 4096),  # 6*6*256=9216 → 4096
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),  # 1줄 → BatchNorm1d

            nn.Linear(4096, 1000),  # 4096 → 1000
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1000),  # 1줄 → BatchNorm1d

            nn.Linear(1000, 100)  # 1000 → 100
        )

    def forward_once(self, x):
        output = self.ConvolutionalLayer(x)
        output = output.view(output.size()[0], -1)
        output = self.FullyConnectedLayer(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# 4. Contrastive Loss 함수 정의
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # min=0.0은 0 미만의 값을 사용하지 않을거라는 의미. → max 함수의 결과가 0인 것과 같음.

        return loss_contrastive

# 5. maching 함수 정의
def matching(img1, img2, network, device):  # img pair를 입력으로 받아서 L2 distance 계산
    transform = transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
    img1 = Image.open(img1)
    img2 = Image.open(img2)
    img1_ = img1.convert("L")
    img2_ = img2.convert("L")  # grayscale로 변환
    img1 = transform(img1_)
    img2 = transform(img2_)
    img1 = torch.unsqueeze(img1, 0)
    img2 = torch.unsqueeze(img2, 0)

    output1, output2 = network(Variable(img1).to(device), Variable(img2).to(device))  # Tensor

    output_vec1 = np.array(output1.cpu().detach().numpy())
    output_vec2 = np.array(output2.cpu().detach().numpy())
    euclidean_distance = np.sqrt(np.sum(np.square(np.subtract(output_vec1, output_vec2))))

    return euclidean_distance

# 6. PyTorch와 ONNX 모델 비교
def compare_two_array(actual, desired, layer_name, rtol=1e-7, atol=0):
    # Reference : https://gaussian37.github.io/python-basic-numpy-snippets/
    flag = False
    try:
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
        print(layer_name + ": no difference.")
    except AssertionError as msg:
        print(layer_name + ": Error.")
        print(msg)
        flag = True
    return flag

# 7. 데이터셋 사용
folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose(
                                            [transforms.Resize((105, 105)), transforms.ToTensor()]),
                                        should_invert=False)

# 8. 학습
train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              num_workers=0,
                              batch_size=Config.train_batch_size)

torch_model = SiameseNetwork().to(device)
optimizer = optim.RMSprop(torch_model.parameters(), lr=1e-5)
criterion = ContrastiveLoss()

counter = []
iteration_number = 0
best_epoch = 0
best_loss = 100.0
best_model = torch_model

for i, data in enumerate(train_dataloader, 0):
    img0, img1, label, _, _, _, _ = data

for epoch in range(0, Config.train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label, _, _, _, _ = data
        img0, img1, label = img0.to(device), img1.to(device), label.to(device)
        optimizer.zero_grad()
        output1, output2 = torch_model(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if best_loss > loss_contrastive.item():
            best_epoch = epoch
            best_loss = loss_contrastive.item()
            best_model = torch_model
        if i % 10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)

end_train_time = time.time()
training_time = end_train_time - start_train_time
print("Total Training Time: %f s" % training_time)

# 9. 매칭 (distance 계산)
start_matching_time = time.time()

gn = 0  # genuine matching 횟수
In = 0  # imposter matching 횟수

# Genuine matching data loader
path = Config.testing_dir
folder_list = os.listdir(path)
file_list = []

for folder_num in range(len(folder_list)):
    print("folder: %s" % folder_list[folder_num])
    dirs = path + '/' + folder_list[folder_num]
    for file in os.listdir(dirs):  # 폴더 안 파일
        file_list.append((dirs + '/' + file, folder_list[folder_num]))  # 폴더 안 (파일 경로,폴더 이름) 튜플 리스트에 저장

combination = list(itertools.combinations(file_list, 2))  # 한 폴더 안 Genuine matching 조합 경우의 수
print(len(combination))  # 첫번째 genuine matching 이미지 pair

torch_model.eval()  # 성능 테스트를 할 때 네트워크 모델을 평가용으로 사용(학습X)

distances_list = []  # squared L2 distance between pairs
identical_list = []  # 1 if same identity, 0 otherwise

In_bal = 0

for c in combination:
    if c[0][1] == c[1][1]:  # genuine matching
        distance = matching(c[0][0], c[1][0], torch_model, device)  # float
        distances_list.append(distance)
        identical_list.append(1)
        gn += 1
        if gn % 100 == 1:
            print("%dth genuine matching..." % gn)
    else:  # imposter matching
        In_bal = In_bal + 1
        if In_bal % Config.test_balance == 1:  # validation, test: 140
            distance = matching(c[0][0], c[1][0], torch_model, device)
            distances_list.append(distance)
            identical_list.append(0)
            In += 1
            if In % 100 == 1:
                print("%dth imposter matching..." % In)

end_matching_time = time.time()
matching_time = end_matching_time - start_matching_time

print("\ngenuine matching 횟수: %d" % gn)
print("imposter matching 횟수: %d" % In)
print("-------------------------\n")
print("Total Matching Time: %f s" % matching_time)

# 10. Threshold 계산
distances_list = np.array(distances_list)
identical_list = np.array(identical_list)

thresholds = np.arange(0.00, 10.0, 0.01)
f1_scores = [f1_score(identical_list, distances_list < t) for t in thresholds]
acc_scores = [accuracy_score(identical_list, distances_list < t) for t in thresholds]

opt_idx = np.argmax(f1_scores)  # Threshold at maximal F1 score
opt_tau = thresholds[opt_idx]
opt_acc = accuracy_score(identical_list, distances_list < opt_tau)  # Accuracy at maximal F1 score

dist_pos = distances_list[identical_list == 1]
dist_neg = distances_list[identical_list == 0]

tpr_list = np.zeros(thresholds.shape)
fpr_list = np.zeros(thresholds.shape)
fnr_list = np.zeros(thresholds.shape)
tnr_list = np.zeros(thresholds.shape)

for i in range(0, len(thresholds)):
    tpr = dist_pos[dist_pos < thresholds[i]]
    tpr = len(tpr) / len(dist_pos)
    tpr_list[i] = tpr
    fnr_list[i] = 1.0 - tpr

    fpr = dist_neg[dist_neg < thresholds[i]]
    fpr = (len(fpr) / len(dist_neg))
    fpr_list[i] = fpr
    tnr_list[i] = 1.0 - fpr

print("f1_scores: %.5f\n" % np.max(f1_scores))

print("Threshold: %.2f\n" % opt_tau)
print("Accuracy: %.5f\n" % opt_acc)

print("True Positive Rate: %.5f" % tpr_list[opt_idx])
print("False Negative Rate: %.5f\n" % fnr_list[opt_idx])

print("False Positive Rate: %.5f" % fpr_list[opt_idx])
print("True Negative Rate: %.5f\n" % tnr_list[opt_idx])

# 11. Pytorch 모델 저장
model_dir = './weights/' + target + ' epoch-{} loss-{} th-{}.pth'.format(best_epoch, best_loss, opt_tau)
torch.save(best_model, model_dir)
print('Best weight: epoch-{} loss-{}'.format(best_epoch, best_loss))

# 12. ONNX 모델로 변환
batch_size = 1    # 임의의 수
torch_model.to('cpu')

# 모델에 대한 입력값
x1 = torch.randn(batch_size, 1, 105, 105, requires_grad=True)
x2 = torch.randn(batch_size, 1, 105, 105, requires_grad=True)
torch_out = torch_model(x1, x2)

# 모델 변환
torch.onnx.export(torch_model,               # 실행될 모델
                  (x1, x2),                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "./onnx/"+target+".onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
                  input_names = ['input1', 'input2'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

# 13. 변환한 ONNX 모델 불러오기
onnx_model = onnx.load("./onnx/"+target+".onnx")

# onnx 모델의 정보를 layer 이름 : layer값 기준으로 저장
onnx_layers = dict()
for layer in onnx_model.graph.initializer:
    onnx_layers[layer.name] = numpy_helper.to_array(layer)

# torch 모델의 정보를 layer 이름 : layer값 기준으로 저장
torch_layers = {}
for layer_name, layer_value in torch_model.named_modules():
    torch_layers[layer_name] = layer_value

# onnx와 torch 모델의 성분은 1:1 대응이 되지만 저장하는 기준이 다르므로
# onnx와 torch의 각 weight가 1:1 대응이 되는 성분만 필터합니다.
onnx_layers_set = set(onnx_layers.keys())
# onnx 모델의 각 layer에는 .weight가 suffix로 추가되어 있어서 문자열 비교 시 추가함
torch_layers_set = set([layer_name + ".weight" for layer_name in list(torch_layers.keys())])
filtered_onnx_layers = list(onnx_layers_set.intersection(torch_layers_set))

# compare_two_array 함수를 통하여 onnx와 torch의 각 대응되는 layer의 값을 비교합니다.
for layer_name in filtered_onnx_layers:
    onnx_layer_name = layer_name
    torch_layer_name = layer_name.replace(".weight", "")
    onnx_weight = onnx_layers[onnx_layer_name]
    torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
    compare_two_array(onnx_weight, torch_weight, onnx_layer_name)

end_running_time = time.time()
running_time = end_running_time - start_running_time

print('Best weight: epoch-{} loss-{}'.format(best_epoch, best_loss))
print("f1_scores: %.5f\n" % np.max(f1_scores))
print("Threshold: %.2f\n" % opt_tau)
print("Accuracy: %.5f\n" % opt_acc)
print("True Positive Rate: %.5f" % tpr_list[opt_idx])
print("False Negative Rate: %.5f\n" % fnr_list[opt_idx])
print("False Positive Rate: %.5f" % fpr_list[opt_idx])
print("True Negative Rate: %.5f\n" % tnr_list[opt_idx])
print("Total Training Time: %f s" % training_time)
print("Total Matching Time: %f s" % matching_time)
print("Total Running Time: %f s" % running_time)
