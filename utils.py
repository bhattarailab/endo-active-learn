import torch
import numpy as np


def cal_rmse(ground_truth, prediction):
    rmse = (ground_truth - prediction) ** 2
    rmse = np.sqrt(rmse.mean())
    return rmse

def cal_mae(ground_truth, prediction):
    mae = np.mean(np.abs(ground_truth - prediction))
    return mae

def relative_mae(ground_truth, prediction):
    return np.mean(np.abs((ground_truth-prediction)/ground_truth))


def calculate_mean_features(features):
    mean_features = torch.zeros_like(features[0]).cuda()
    for x in features:
        mean_features+=x
    return mean_features/len(features)

def calculate_variance_features(features, mean_feature):
  sum_s = torch.zeros_like(mean_feature).cuda()
  for x in features:
    sum_s += torch.square(x - mean_feature)
  return sum_s/len(features)


class MeanIOU():
    def __init__(self, num_classes=2, threshold=0.5):
        self.num_classes = 2
        self.threshold = threshold
        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int64)
        self.reset()

    def reset(self):
        self.conf.fill(0)
        
    def add(self, predicted, target):
         # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 4 or predicted.dim() == 3, \
            "predictions must be of dimension (N, 1, H, W) or (1, H, W)"
        assert target.dim() == 4 or target.dim() == 3, \
            "targets must be of dimension (N, 1, H, W) or (1, H, W)"
        
        #Convert predictions into foreground and background according the threshold
        predicted = (predicted >= self.threshold).long().view(-1).numpy()
        target = (target >= self.threshold).long().view(-1).numpy()
       
        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
                    x.astype(np.int64), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))
        
        self.conf +=conf
        
    def value(self):   
        # Extract different metrics in confidence matrix
        true_positive = np.diag(self.conf)
        false_positive = np.sum(self.conf, 0) - true_positive
        false_negative = np.sum(self.conf, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        
        return np.nanmean(iou)
        
        
        