from cmath import nan
import numpy as np
import torch 
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
import os
# import cv2
from PIL import Image

def make_data_loader(args):
    if args.dataset == 'kaggle':
       
        pth1='./weizmann_horse_db/horse_train'
        pth2='./weizmann_horse_db/mask_train'
        testpth1='./weizmann_horse_db/horse_test'
        testpth2='./weizmann_horse_db/mask_test'
        transform_img = transforms.Compose([
        transforms.Resize((80, 100), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
    ])
        transform_label= transforms.Compose([
        transforms.Resize((80, 100), interpolation=Image.NEAREST),
    ])

        train_dataset = MyDataset(input_root=pth1,label_root=pth2,transform_img=transform_img,transform_label=transform_label)
        test_dataset = MyDataset(input_root=testpth1,label_root=testpth2,transform_img=transform_img,transform_label=transform_label)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        num_class = 2

        return train_loader, test_loader, num_class
    else:
        raise NotImplementedError

class MyDataset(Dataset):#继承了Dataset子类
    def __init__(self,input_root,label_root,transform_img=None,transform_label=None):
        #分别读取输入/标签图片的路径信息
        self.input_root=input_root
        self.input_files=os.listdir(input_root)#列出指定路径下的所有文件
        
        self.label_root=label_root
        self.label_files=os.listdir(label_root)
 
        self.transforms_img=transform_img
        self.transforms_label=transform_label
    def __len__(self):
        #获取数据集大小
        return len(self.input_files)
    def __getitem__(self, index):
        #根据索引(id)读取对应的图片
        input_img_path=os.path.join(self.input_root,self.input_files[index])
        input_img=Image.open(input_img_path).convert('RGB')
        input_img=self.transforms_img(input_img)
 
        label_img_path=os.path.join(self.label_root,self.label_files[index])
        label_img=Image.open(label_img_path)
        label_img=self.transforms_label(label_img)
        
        label_img =np.array(label_img)
        label_img = torch.FloatTensor(label_img)
 
        return (input_img,label_img)

if __name__=='__main__':
    # train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=4000, #20000,  #AAAAA
    #                                                                     num_test_samples=1000,#5000,
    #                                                                     image_shape=(64, 64),
    #                                                                     max_num_digits_per_image=4,
    #                                                                     num_classes=10)
    # # ->(B,C,W,H)
    # train_x = train_x.reshape(train_x.shape[0],train_x.shape[3],train_x.shape[2],train_x.shape[1])

    # train_x = np.append(train_x,np.zeros([train_x.shape[0],2,train_x.shape[2],train_x.shape[3]]),axis=1)

    # # train_y = train_y.reshape(train_y.shape[0],train_y.shape[3],train_y.shape[2],train_y.shape[1])
    # # re_train_y = np.zeros([train_y.shape[0],train_y.shape[2],train_y.shape[3]])
    # re_train_y = np.full([train_y.shape[0],train_y.shape[2],train_y.shape[2]],10)   #4<<<---
    # for u in range(train_y.shape[0]):
    #     for i in range(train_y.shape[1]):
    #         for j in range(train_y.shape[2]):               
    #             for k in range(10):    
    #                 if(train_x[u,0,i,j]==0):
    #                     break               
    #                 if(train_y[u,i,j,k]==train_x[u,0,i,j]):
    #                     re_train_y[u,i,j]=k
    #                     break
                


    # test_x = test_x.reshape(test_x.shape[0],test_x.shape[3],test_x.shape[2],test_x.shape[1])
    # test_x = np.append(test_x,np.zeros([test_x.shape[0],2,test_x.shape[2],test_x.shape[3]]),axis=1)
    
    # # test_y = test_y.reshape(test_y.shape[0],test_y.shape[3],test_y.shape[2],test_y.shape[1])

    # # re_test_y = np.zeros([test_y.shape[0],test_y.shape[2],test_y.shape[3]])
    # re_test_y = np.full([test_y.shape[0],test_y.shape[2],test_y.shape[2]],10)
    # for u in range(test_y.shape[0]):
    #     for i in range(test_y.shape[1]):
    #         for j in range(test_y.shape[2]):                
    #             for k in range(10):      
    #                 if(test_x[u,0,i,j]==0):
    #                     break            
    #                 if (test_y[u,i,j,k]==test_x[u,0,i,j]):
    #                     re_test_y[u,i,j]=k
    #                     break
    # pth1='D:/code/数据集/MINIST_EXTENDED/4000_0_9/train_x.npy'
    # pth11='D:/code/数据集/MINIST_EXTENDED/4000_0_9/train_x归一化0_1.npy'
    # pth2='D:/code/数据集/MINIST_EXTENDED/4000_0_9/train_y.npy'
    # pth22='D:/code/数据集/MINIST_EXTENDED/4000_0_9/train_y_BHWC.npy'
    # pth3='D:/code/数据集/MINIST_EXTENDED/4000_0_9/test_x.npy'
    # pth33='D:/code/数据集/MINIST_EXTENDED/4000_0_9/test_x归一化0_1.npy'
    # pth4='D:/code/数据集/MINIST_EXTENDED/4000_0_9/test_y.npy'
    # pth44='D:/code/数据集/MINIST_EXTENDED/4000_0_9/test_y_BHWC.npy'

    # pth1='D:/code/数据集/MINIST_EXTENDED/3840_0_3/train_x123.npy'
    # pth2='D:/code/数据集/MINIST_EXTENDED/3840_0_3/train_y123.npy'
    # pth22='D:/code/数据集/MINIST_EXTENDED/3840_0_3/train_yr_10ch123.npy'
    # pth3='D:/code/数据集/MINIST_EXTENDED/3840_0_3/test_x123.npy'
    # pth4='D:/code/数据集/MINIST_EXTENDED/3840_0_3/test_y123.npy'
    # pth44='D:/code/数据集/MINIST_EXTENDED/3840_0_3/test_y_10ch123.npy'

    # pth11='D:/code/数据集/MINIST_EXTENDED/train_x1233.npy'
    # pth111='D:/code/数据集/MINIST_EXTENDED/train_x1233_256.npy'
    # pth21='D:/code/数据集/MINIST_EXTENDED/train_y1233.npy'
    # pth221='D:/code/数据集/MINIST_EXTENDED/train_yr_10ch1233.npy'
    # pth31='D:/code/数据集/MINIST_EXTENDED/test_x1233.npy'
    # pth311='D:/code/数据集/MINIST_EXTENDED/test_x1233_256.npy'
    # pth41='D:/code/数据集/MINIST_EXTENDED/test_y1233.npy'
    # pth441='D:/code/数据集/MINIST_EXTENDED/test_y_10ch1233.npy'
    # print(train_x.shape)
    # train_x=train_x[:,0,:,:]
    # print(train_x.shape)
    # train_x=train_x.reshape([4000,1,64,64])
    # train_x=np.repeat(train_x, 3, axis=1)
    # np.save(pth11,train_x)
    # train_x=train_x*255
    # print(train_x.shape)

    # print(test_x.shape)
    # test_x=test_x[:,0,:,:]
    # print(test_x.shape)
    # test_x=test_x.reshape([1000,1,64,64])
    # test_x=np.repeat(test_x, 3, axis=1)
    # np.save(pth33,test_x)
    # test_x=test_x*255
    # print(test_x.shape)

    # np.save(pth1,train_x)
    # np.save(pth2,re_train_y)
    # np.save(pth22,train_y)

    # np.save(pth3,test_x)
    # np.save(pth4,re_test_y)
    # np.save(pth44,test_y)
    print('ok')
    # pth1='D:/code/数据集/MINIST_EXTENDED/train_x.npy'
    # pth2='D:/code/数据集/MINIST_EXTENDED/train_y.npy'
    # pth22='D:/code/数据集/MINIST_EXTENDED/train_yr_10ch.npy'
    # pth3='D:/code/数据集/MINIST_EXTENDED/test_x.npy'
    # pth4='D:/code/数据集/MINIST_EXTENDED/test_y.npy'
    # pth44='D:/code/数据集/MINIST_EXTENDED/test_y_10ch.npy'
    # ->(B,C,W,H)
    # train_x = np.load(pth11)
    # re_train_y = np.load(pth2)
    # test_x = np.load(pth31)
    # re_test_y = np.load(pth4)
    # train_y=np.load(pth22)
    # print(train_x.shape)
    # train_x=train_x[:,0,:,:]
    # test_x=test_x[:,0,:,:]
    # print(train_x.shape)
    

