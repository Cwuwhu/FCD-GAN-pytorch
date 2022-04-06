import numpy as np

# Accuracy assessment
# https://github.com/I-Hope-Peace/ChangeDetectionRepository

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Kappa(self):
        Po = self.Pixel_Accuracy()
        Pe = np.dot(self.confusion_matrix.sum(axis=0), self.confusion_matrix.sum(axis=1)) / np.square(self.confusion_matrix.sum())
        kappa = (Po - Pe) / (1 - Pe)
        return kappa

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        mAcc = np.nanmean(Acc)
        return mAcc, Acc

    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Pre = self.confusion_matrix[1, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        return Pre

    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return Rec

    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = 2 * Rec * Pre / (Rec + Pre)
        return F1

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        cIoU = MIoU[1].copy()
        MIoU = np.nanmean(MIoU)
        return MIoU, cIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def _generate_matrix_bymap(self, gt_image, pre_image, gt_map=[0, 1], pre_map=[0, 1]):
        confusion_matrix = np.zeros((self.num_class,) * 2)
        for i in range(len(gt_map)):
            for j in range(len(pre_map)):
                confusion_matrix[i, j] = np.sum((gt_image == gt_map[i]) & (pre_image == pre_map[j]))
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def add_batch_map(self, gt_image, pre_image, gt_map=[0, 1], pre_map=[0, 1]):
        assert gt_image.shape == pre_image.shape
        assert len(gt_map) == len(pre_map)
        assert len(gt_map) == self.num_class
        self.confusion_matrix += self._generate_matrix_bymap(gt_image, pre_image, gt_map=gt_map, pre_map=pre_map)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
