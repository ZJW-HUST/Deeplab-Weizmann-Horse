import numpy as np
import torch
import cv2


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.intersection=0
        self.union=0

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        print('confusion matrix:\n',self.confusion_matrix) #change1
        print('sum',self.confusion_matrix.sum())
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
    def biou(self):
        return self.intersection/self.union

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        '''
        输入为ndarray
        gt_image：(batch_size,H,W)
        pre_image：(batch_size,H,W)
        '''
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        a,b = self.boundary_iou(gt=gt_image,dt=pre_image)
        self.intersection += a
        self.union += b 

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.intersection=0
        self.union=0
    
    def mask_to_boundary(self,mask, dilation_ratio=0.02, sign=1):
        mask = mask.astype('uint8')
        b, h, w = mask.shape
        new_mask = np.zeros([b, h + 2, w + 2])
        mask_erode = np.zeros([b, h, w])
        img_diag = np.sqrt(h ** 2 + w ** 2)  # 计算图像对角线长度
        # 计算腐蚀的程度dilation
        dilation = int(round(dilation_ratio * img_diag))
        if dilation < 1:
            dilation = 1

        # 对一个batch中所有进行腐蚀操作
        for i in range(b):
            new_mask[i] = cv2.copyMakeBorder(mask[i], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # 用0填充边框

        kernel = np.ones((3, 3), dtype=np.uint8)
        for j in range(b):
            # 腐蚀操作
            new_mask_erode = cv2.erode(new_mask[j], kernel, iterations=dilation)
            # 回填
            mask_erode[j] = new_mask_erode[1: h + 1, 1: w + 1]

        return mask - mask_erode

    # 获取标签和预测的边界iou
    def boundary_iou(self,gt, dt, dilation_ratio=0.1):
        dt_boundary = self.mask_to_boundary(dt, dilation_ratio, sign=1)
        gt_boundary = self.mask_to_boundary(gt, dilation_ratio, sign=0)
        B, H, W = dt_boundary.shape
        intersection = 0
        union = 0
        # 计算交并比
        for k in range(B):
            intersection += ((gt_boundary[k] * dt_boundary[k]) > 0).sum()
            union += ((gt_boundary[k] + dt_boundary[k]) > 0).sum()
        if union < 1:
            return 0
        boundary_iou = intersection / union

        return intersection,union
        # return boundary_iou





