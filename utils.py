import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(phase: str):
    if phase == 'train':
        return A.Compose([
            A.RandomResizedCrop(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.OneOf([
            A.RandomBrightnessContrast(p=0.5),A.RandomGamma(p=0.5)], p=0.5),
            A.OneOf([
            A.Blur(p=0.4),
            A.GaussianBlur(p=0.4),
            A.MotionBlur(p=0.4)], p=0.4),
            A.OneOf([
            A.GaussNoise(p=0.4),
            A.ISONoise(p=0.4),
            A.GridDropout(ratio=0.5, p=0.2),
            A.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, p=0.2)], p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(),
            ToTensorV2(),
        ])


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


