from utils import *
import matplotlib.pyplot as plt
import os
import shutil
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class TrainSetLoader(Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset_name,
        patch_size,
        img_norm_cfg=None,
        return_meta=False,
        reliability_label_dir=None,
        return_reliability=False,
    ):
        super(TrainSetLoader).__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        self.return_meta = bool(return_meta)
        self.reliability_label_dir = reliability_label_dir
        self.return_reliability = bool(return_reliability)
        if not os.path.exists(self.dataset_dir +'/img_idx/train_' + dataset_name + '.txt') and os.path.exists(self.dataset_dir +'/img_idx/train.txt'):
            shutil.copyfile(self.dataset_dir +'/img_idx/train.txt', self.dataset_dir +'/img_idx/train_' + dataset_name + '.txt')
        with open(self.dataset_dir +'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def _load_reliability_npz(self, image_id):
        if not self.return_reliability:
            return None, None, None
        if self.reliability_label_dir is None:
            raise ValueError("return_reliability=True but reliability_label_dir is None")
        path = os.path.join(self.reliability_label_dir, image_id + '.npz')
        if not os.path.exists(path):
            raise FileNotFoundError("Missing reliability pseudo label: %s" % path)
        data = np.load(path)
        rel_label = data['rel_label'].astype(np.float32)
        rel_valid = data['rel_valid'].astype(np.float32)
        if 'tce_prob' in data:
            tce_prob = data['tce_prob'].astype(np.float32)
        else:
            tce_prob = np.zeros_like(rel_label, dtype=np.float32)
        return rel_label, rel_valid, tce_prob
        
    def __getitem__(self, idx):
        image_id = self.train_list[idx]
        try:
            img = Image.open((self.dataset_dir + '/images/' + image_id + '.png').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + image_id + '.png').replace('//','/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + image_id + '.bmp').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + image_id + '.bmp').replace('//','/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]

        rel_label, rel_valid, tce_prob = self._load_reliability_npz(image_id)
        img_patch, mask_patch, coords = random_crop_with_coords(img, mask, self.patch_size, pos_prob=0.5)
        if self.return_reliability:
            rel_label_patch = crop_by_coords(rel_label, coords, patch_size=self.patch_size)
            rel_valid_patch = crop_by_coords(rel_valid, coords, patch_size=self.patch_size)
            tce_prob_patch = crop_by_coords(tce_prob, coords, patch_size=self.patch_size)
        else:
            rel_label_patch = rel_valid_patch = tce_prob_patch = None

        img_patch, mask_patch, aug_ops = self.tranform(img_patch, mask_patch, return_ops=True)
        if self.return_reliability:
            rel_label_patch = apply_aug_ops(rel_label_patch, aug_ops)
            rel_valid_patch = apply_aug_ops(rel_valid_patch, aug_ops)
            tce_prob_patch = apply_aug_ops(tce_prob_patch, aug_ops)
        img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        if self.return_reliability:
            rel_label_patch = torch.from_numpy(np.ascontiguousarray(rel_label_patch[np.newaxis, :]))
            rel_valid_patch = torch.from_numpy(np.ascontiguousarray(rel_valid_patch[np.newaxis, :]))
            tce_prob_patch = torch.from_numpy(np.ascontiguousarray(tce_prob_patch[np.newaxis, :]))
            if self.return_meta:
                return (
                    img_patch,
                    mask_patch,
                    rel_label_patch,
                    rel_valid_patch,
                    tce_prob_patch,
                    image_id,
                    torch.tensor(aug_ops, dtype=torch.long),
                )
            return img_patch, mask_patch, rel_label_patch, rel_valid_patch, tce_prob_patch
        if self.return_meta:
            return img_patch, mask_patch, image_id, torch.tensor(aug_ops, dtype=torch.long)
        return img_patch, mask_patch
    def __len__(self):
        return len(self.train_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//','/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//','/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        
        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)
        
        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 

class InferenceSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(InferenceSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        
        h, w = img.shape
        img = PadImg(img)
        
        img = img[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        return img, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 


class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        mask_pred = Image.open((self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + '.png').replace('//','/'))
        mask_gt = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')

        mask_pred = np.array(mask_pred, dtype=np.float32)  / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32)  / 255.0
        
        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:,:,0]
        
        h, w = mask_pred.shape
        
        mask_pred, mask_gt = mask_pred[np.newaxis,:], mask_gt[np.newaxis,:]
        
        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h,w]
    def __len__(self):
        return len(self.test_list) 


class augumentation(object):
    def __call__(self, input, target, return_ops=False):
        ops = [0, 0, 0]
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
            ops[0] = 1
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
            ops[1] = 1
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
            ops[2] = 1
        if return_ops:
            return input, target, ops
        return input, target
