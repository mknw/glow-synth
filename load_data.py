#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF
import torch.backends.cudnn as cudnn

from skimage import io
from PIL import Image, ImageDraw
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import random  


def sample_celeba(batch, image_size, test=False):
    if not test:
        split = 'train'
        shuffle = True
        transform = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Resize(size=image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])
    else:
        split = 'test'
        transform = transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])
    print(f'shuffle set to {shuffle} for split: {split}')
    target_type = ['attr', 'bbox', 'landmarks']
    dataset = datasets.CelebA(root='data', split=split, target_type=target_type[0], download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=16)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=16)
            loader = iter(loader)
            yield next(loader)

def sample_from_directory(path, batch, image_size, test=False, shuffle=False):
    seed = 2147483647
    torch.manual_seed(seed)

    if not test:
        # train split
        transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.Resize(size=image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])
    else:
        # test split
        transform = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])

    dataset = datasets.ImageFolder(root = path, transform=transform)
    if not test:
        dataset, _ = random_split(dataset, [65000, 5000])
    else: 
        _, dataset= random_split(dataset, [65000, 5000])

    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=16)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=16)
            loader = iter(loader)
            yield next(loader)


def sample_FFHQ_eyes(batch, image_size, degrees=90, transform=None, shuffle=True):
    if transform is None:
        transform = transforms.Compose([
                    RandomRotatedResizedCrop(output_size=image_size),
                    transforms.RandomRotation(degrees),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
                ])
    dataset = FFHQLandmarks(transform=transform)
    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=16)
    loader = iter(loader)
    
    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=16)
            loader = iter(loader)
            yield next(loader)



class FFHQLandmarks(Dataset):

    def __init__(self, json_file='ffhq-dataset-v2.json', root_dir='data/FFHQ',
                         target_attributes = ['category', 'image'], transform=None):

        assert set(target_attributes).issubset(['category', 'image', 'thumbnail', 'eyes'])
        
        self.root_dir = root_dir
        self.transform = transform
        if not transform:
            self.totensor = transforms.ToTensor()

        self.target_attributes = target_attributes
        metadata_frame = pd.read_json(os.path.join(root_dir, json_file), orient='index')
        self.df = self.format_dataframe(metadata_frame, target_attributes)

    def format_dataframe(self, metadata, target_attributes):
        assert target_attributes[0] == 'category'
        df = pd.DataFrame(metadata.iloc[:,0].tolist())
        df.columns = ['split']

        for target in target_attributes[1:]:
            df_temp = pd.DataFrame(metadata[target].tolist())
            df_temp.columns = [f'{target[:2]}_{c}' for c in df_temp.columns]
            df = pd.concat([df, df_temp], axis=1)
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.df.im_file_path[idx])
        image = Image.open(img_name)

        eyes_lms = self.eye_landmarks(idx)
        # sample = {'image': image, 'landmarks': eyes_lms}
        if self.transform:
            image, eyes_lms = self.transform((image, eyes_lms))
        else:
            image = self.totensor(image)
            # return (image, eyes_lms, img_name)
        return (image, eyes_lms)

    def eye_landmarks(self, idx):
        eyes_lms = np.array((self.df.iloc[idx]['im_face_landmarks'][36:42],
                    self.df.iloc[idx]['im_face_landmarks'][42:48]))
        return eyes_lms



class RandomRotatedResizedCrop(object):
    ''' 
    Args:
        ratio (int): Proportion of output image. E.g.: 0.5 for up to half 
        output_size (int): Desired output size in pixels.
    '''

    def __init__(self, output_size=256, crop_margin = 24, degrees=90):
        self.output_size = output_size # 256
        self.crop_margin = crop_margin
        self.min_eyes_h = 20
        self.num_rotations = (3, 5)
        self.concentric_reps = 3

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
        # print(f'PARAMS: crop_margin: {crop_margin}, circular reps: {self.num_rotations}', end='')

    def __call__(self, input):

        bg = False
        input_image, input_eyes = input
        inner_h = None
        rot_params = None
        in_deg = None

        images = []; masks = []; eyes_l = []

        for cr in range(self.concentric_reps):

            # copy and pad original image, take bbox coordinates
            image, eyes, ratio, inner_h = self._preproc_resize(
                                           input_image.copy(), input_eyes.copy(), inner_h)

            base_image, eye_vertices, eyes, in_deg = self._init_rotate(
                                           image.copy(), eyes, ratio, in_deg, rot_params)

            base_image, mask, rot_params = self._rotate_and_composite_n(
                                           base_image, eye_vertices, bg, rot_params)

            images.append(base_image.copy())
            masks.append(mask.copy())
            eyes_l.append(eyes.copy())

        # crop background
        base_image = images[0].copy()
        fg_mask = masks[0].copy()
        for i in range(self.concentric_reps-1):
            bg_image = self.crop_image(images[i+1])
            base_image = Image.composite(base_image, bg_image, fg_mask)

            # if not i == self.concentric_reps-2:
            fg_mask = Image.composite(fg_mask, self.crop_image(masks[i+1]), fg_mask)

        c = self.random_color()
        base_image = Image.composite(base_image, Image.new('RGB', base_image.size, color=c), fg_mask)

        # add 1 eye
        image = self.add_eye(base_image, images[0], eyes_l)

        # tensorization
        image = self.to_tensor(image) # mask.convert('RGB')) 
        image = self.normalize(image)
        return image, eyes_l[0]
    

    def add_eye(self, base_image, source_image, eyes):
        ''' add central eye '''

        eye = eyes[0][random.randint(0, 1)].copy()

        eye_center = np.array((eye[:, 0].min() + eye[:,0].max(),
                               eye[:, 1].min() + eye[:,1].max())) / 2
        m = 4

        x_eye = [x + m if x > eye_center[0] else x - m for x in eye[:,0]]
        y_eye = [y + m if y > eye_center[1] else y - m for y in eye[:,1]]
        eye = np.stack([x_eye, y_eye], axis=1)

        eye_bb = (eye[:, 0].min(), eye[:,1].min(),
                  eye[:, 0].max(), eye[:,1].max())

        eye_crop = source_image.crop(eye_bb)
        eye[:, 0] -= eye_bb[0]
        eye[:, 1] -= eye_bb[1]
        # eye_center[0] -= eye_bb[0]
        # eye_center[1] -= eye_bb[1]
        # paste_xy = (128 - eye_center).astype(np.int8)

        eye_mask = Image.new('L', size=eye_crop.size)
        mask_draw = ImageDraw.Draw(eye_mask)
        mask_draw.polygon([tuple(c) for c in eye], fill=255)

        eye_crop.putalpha(eye_mask)
        
        # rotate eye (eye roll)
        deg = random.randrange(-90, 90)
        eye_crop = eye_crop.rotate(deg, expand=True)

        paste_xy = tuple([int(128 - coord / 2) for coord in eye_crop.size])

        base_image = base_image.convert('RGBA')
        base_image.paste(eye_crop, paste_xy, eye_crop)
        return base_image.convert('RGB')


    def random_color(self):
        levels = range(32, 256, 32)
        return tuple(random.choice(levels) for _ in range(3))
    
    def bbox_measures(self, eyes, side):
        min_x, min_y, max_x, max_y = self.take_bbox(eyes, self.crop_margin)
        eyes_crop_h = max_y - min_y
        eyes_crop_w = max_x - min_x
        diag = self.output_size * sqrt(2)

        ## Resize by giving crop height.
        # max_h: H for crop with W spanning 256 px
        max_h = int(self.output_size * (eyes_crop_h / eyes_crop_w))
        # diag_h: H for crop with W spanning `diag` pixels (~ 362).
        diag_h = int(diag * (eyes_crop_h / eyes_crop_w))

        max_inner = int(max_h / 2)
        max_outer = int(diag_h / self.concentric_reps)
        crop_input = (min_y, min_x, eyes_crop_h, eyes_crop_w)
        # return (max_inner, max_outer, crop_input)
        return (max_inner, max_outer, crop_input)

    def _preproc_resize(self, image, eyes, inner_h=None):
        ''' 
        Crops area around eyes; resizes with smaller in range (self.crop_margin, max_h).
        Args:
            input (tuple): (PIL.Image, np.ndarray)
        '''

        max_inner, max_outer, crop_input = self.bbox_measures(eyes, image.height)

        if not inner_h: 
            if max_inner < self.min_eyes_h:
                H = max_inner
            else:
                H = random.randint(self.min_eyes_h, max_inner) # + self.crop_margin)
        else:
            # = random.randint(max_inner, max_outer) # + self.crop_margin)
            H = inner_h * 2 # + 20

        image = TF.resized_crop(image.copy(), *crop_input, H) # , min_y, min_x, eyes_crop_h, eyes_crop_w, random_h)
        # update landmarks coordinates against
        new_w, new_h = image.size       # original size:
        eyes[:,:,0] -= crop_input[1]    # <- min_x
        eyes[:,:,1] -= crop_input[0]    # <- min_y
        ratio = new_h / crop_input[2]   # <- eyes_crop_h
        eyes *= ratio # and resize 
        return (image, eyes, ratio, H)


    def _adjust_silhuette(self, bbox_vrtc, eyes):
        left_vtx = (bbox_vrtc[0,0], eyes[0,0,1])
        right_vtx = (bbox_vrtc[1, 0], eyes[1, 3, 1])
        insert_vrtc = np.array([left_vtx, right_vtx])

        bbox_vrtc[0, 0] = eyes[0, -1,0].copy() # bottom-left eyelid
        bbox_vrtc[1, 0] = eyes[1, -2,0].copy() # bottom-right eyelid
        bbox_vrtc[2, 0] = eyes[1, 2, 0].copy()
        bbox_vrtc[3, 0] = eyes[0, 1, 0].copy()
        bbox_vrtc = np.insert(bbox_vrtc, (0, 2), insert_vrtc, axis=0)
        return bbox_vrtc

    def _init_rotate(self, image, eyes, ratio, in_degrees=None, rot_params=None):
        ''' set initial random rotation on image and coordinates.'''
        # pad image, take bbox according to new ratio
        if in_degrees:
            s = max(image.size)
            image, eyes = self.pad_image(image, size=(s, s), eyes=eyes, padding_mode='constant')
        else:
            image, eyes = self.pad_image(image, eyes=eyes, padding_mode='constant')

        bbox_coords = self.take_bbox(eyes,
                           crop_margin = self.crop_margin * ratio)
        bbox_vertices = _find_coords_vertices(*bbox_coords)
        bbox_vertices = self._adjust_silhuette(bbox_vertices, eyes)

        # random initial rotation. 
        if in_degrees:
            if not rot_params: raise ValueError
            in_degrees = in_degrees - (rot_params[1] / 2)
            radians = in_degrees * 3.1415926 / 180
            # image = image.rotate(- in_degrees, expand=False)
            initRotation = RotationCoords(radians,
                               xy_t=np.array(image.size)/2, y_span=s)
        else:
            _, in_degrees, in_radians = self.random_degrees(90)
            initRotation = RotationCoords(in_radians)

        image = image.rotate(- in_degrees) # , expand=True)

        eyes = initRotation(np.array(eyes))
        bbox_vertices = initRotation(np.array(bbox_vertices))
        # TODO bbox_vertices.jjkk

        return (image, bbox_vertices, eyes, in_degrees)

    def _rotate_and_composite_n(self, base_image, bbox_coords, bg=False,
                                  rotation_params = None):

        if not rotation_params:
            reps, degrees, radians = self.random_degrees(*self.num_rotations)
        else:
            reps, degrees, radians = rotation_params

        maskRotation = RotationCoords(radians, xy_t=np.array(base_image.size)/2,
                                        y_span = base_image.height, verbose=False)

        mask = Image.new('1', base_image.size)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.polygon([tuple(c) for c in bbox_coords], fill=255)
        base_image = Image.composite(base_image, mask.copy().convert('RGB'), mask)

        for r in range(reps):
            r_image = base_image.copy().rotate(- degrees * (r + 1), expand=False)
            base_image = Image.composite(base_image, r_image, mask)

            bbox_coords = maskRotation(coords=np.array(bbox_coords))
            draw_mask.polygon([tuple(c) for c in bbox_coords], fill=255, outline=0)

        return (base_image, mask, (reps, degrees, radians))
        

    def random_degrees(self, start, stop=None):
        ''' Args:
            if only start: n degrees rotation in both 
                clockwise and anti-c.w. directions
            if start and stop: n repetitions for half
                circle (180 deg.)'''
        if not stop:
            reps = None
            degrees = random.random() * start
        else:
            reps = random.randrange(start, stop)
            degrees = 180. / (reps + 1)

        if random.random() < .5:
            degrees = -degrees

        radians = degrees * 3.1415926 / 180
        return (reps, degrees, radians)

    def take_bbox(self, landmarks, crop_margin=0):
        min_x = landmarks[:,:,0].min() - crop_margin
        max_x = landmarks[:,:,0].max() + crop_margin
        min_y = landmarks[:,:,1].min() - crop_margin
        max_y = landmarks[:,:,1].max() + crop_margin
        return (min_x, min_y, max_x, max_y)

    def crop_image(self, image, size=None, eyes=None):
        w, h = image.size
        if not size:
            t_h = t_w = self.output_size
        else:
            assert len(size) == 2; 'Two params (h, w) expected for padding'
            t_w, t_h = size
        if h == t_h:
            if eyes is not None:
                return (image, eyes)
            else:
                return image

        left_crop = (w - t_w) // 2
        r_crop = t_w + left_crop
        top_crop = (h - t_h) // 2
        b_crop = t_h + top_crop

        margin = (left_crop, top_crop, r_crop, b_crop)

        image = image.crop(margin)
        if eyes is not None:
        # update landmarks coords
            eyes = eyes.copy()
            eyes[:,:,0] += ( self.output_size - w ) // 2
            eyes[:,:,1] += ( self.output_size - h ) // 2
            return (image, eyes)
        else:
            return image


    def pad_image(self, image, size=None, fill=0, padding_mode='constant', eyes = None):
        w, h = image.size
        if not size:
            t_h = t_w = self.output_size
        else:
            assert len(size) == 2; 'Two params (h, w) expected for padding'
            t_w, t_h = size

        left_pad = (t_w - w) // 2
        r_pad = t_w - (w + left_pad)
        top_pad = (t_h - h) // 2
        b_pad = t_h - (h + top_pad)
        padding = (left_pad, top_pad, r_pad, b_pad)
        image = TF.pad(image, padding, fill, padding_mode)

        if eyes is not None:
        # update landmarks coords
            dims = eyes.shape
            if len(eyes.shape) > 2:
                eyes = eyes.reshape(-1, 2)
            eyes[:,0] += ( t_w - w ) // 2
            eyes[:,1] += ( t_h - h ) // 2
            return (image, eyes.reshape(dims))
        else:
            return image


class RotationCoords():

    def __init__(self, radians, verbose=False, xy_t = np.array((128, 128)), y_span=256):
        if y_span in xy_t:
            raise Warning('translation value for y should be y_span / 2 = {y_span / 2}')
        self.tr_mtx_to_o = self.translation_matrix(xy_t)
        self.y_span = y_span

        self.rot_mtx = self.rotation_matrix(radians)
        self.tr_mtx_to_c = self.translation_matrix(xy_t, to_origin=False)
        self.v = verbose

    def __call__(self, coords, reshape=True):

        if self.v: print(f'Points: {coords}')
        coords_mtx = self._set_affine_tr(coords, reshape)
        if self.v: print(f'Affine transformation matrix: {coords}')
        coords = self.tr_mtx_to_o.dot(coords_mtx.T)
        if self.v: print(f'Origin: {coords}')
        coords = self.rot_mtx.dot(coords)
        if self.v: print(f'Rotation: {coords}')
        coords = self.tr_mtx_to_c.dot(coords).T
        if self.v: print(f'To centre: {coords}')
        # (coords[0, 0], coords[0, 1], coords[0, 1], coords[1, 1])
        coords = self._shift_ordinates_origin(coords)
        return coords[:, :2].reshape(self.coords_shape)
    
    def _shift_ordinates_origin(self, coords):
        coords[:, 1] = np.abs(coords[:, 1] - self.y_span)
        return coords

    def rotation_matrix(self, theta):
        return np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],[0,0,1]])
    
    def translation_matrix(self, xy, to_origin=True):
        transl_mtx= np.eye(3)
        if to_origin:
            transl_mtx[:-1, -1] = -1 * xy
            # transl_mtx[0, -1] *= -1
        else:
            transl_mtx[:-1, -1] = xy # grid back inplace.
        return transl_mtx

    def _set_affine_tr(self, coords, reshape):
        # if isinstance(coords, (list, tuple)) and len(coords) == 4:
        
        if reshape:
            if coords.size == 4:
                coords = _find_coords_vertices(*coords)
            self.coords_shape = coords.shape
            assert coords.shape[-1] == 2
            if len(coords.shape) > 2:
                coords = coords.reshape(-1, 2)

        coords = self._shift_ordinates_origin(coords)
        return np.concatenate([coords, np.ones(
                                 coords.shape[0]).reshape(
                                 coords.shape[0],1)], axis=1)

def _find_coords_vertices(min_x, min_y, max_x, max_y):
    return np.array([[min_x, max_y], [max_x, max_y], [max_x, min_y], [min_x, min_y]])


def show_landmarks(image, landmarks):
    img_s = image.shape[-2:]
    img = np.moveaxis(image.detach().cpu().numpy().reshape(3, *img_s), 0, -1)
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    lms = landmarks.detach().cpu().numpy().reshape(-1, 2)
    plt.scatter(lms[:,0].flatten(), lms[:,1].flatten(), s=10, marker='.', c='r')
    plt.pause(0.001)


def load_network(model_dir, device, conf, checkpoint=True):
    if conf.arch == 'glow':
        from model import Glow
        net = Glow(3, conf.n_flows, conf.n_blocks, affine=conf.affine, conv_lu=True)
        from train_like import calc_loss
        loss_fn = calc_loss
    elif conf.arch in ['densenet', 'resnet']:
        raise NotImplementedError
    torch.cuda.empty_cache()
    net = net.to(device)
    if str(device).startswith('cuda'):
        net = torch.nn.DataParallel(net, conf.gpus)
        cudnn.benchmark = conf.benchmark

    # load checkpoint
    if checkpoint:
        checkpoint = torch.load(model_dir)
        try:
            net.load_state_dict(checkpoint['net'])
        except RuntimeError as re:
            print(re)
            raise ArchError('There is a problem importing the model, check parameters.')

    return net, loss_fn


class Attributes:
    ''' minimal accessory class for correct 
    import/export of dataframe with
    proper column labeling for celeba attributes'''

    def __init__(self):
        self.filename='data/1_den_celeba/attr_y.csv'
        self.headers = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]
        self.colors = None # for now XXX

    def fetch(self, filename='data/1_den_celeba/attr_y.pkl'):
        # make it pandas dataframe
        if filename.endswith('.pkl') or filename.endswith('.pickle'):
            self.df = pd.read_pickle(filename)
        elif filename.endswith('.csv'):
            self.df_csv = pd.read_csv(filename, index_col=False, dtype=np.int8)
            raise NotImplementedError
        print('Fetched dataframe from file {}.'.format(filename))
        return self

    def make_and_save_dataframe(self, y, filename='data/1_den_celeba/attr_y.pkl'):
        df = pd.DataFrame(y, columns=self.headers, dtype=np.int8)
        if filename.endswith('.pkl'):
            df.to_pickle(filename) # this should be it. Nice and simple.
        elif filename.endswith('.csv'):
            warn('deprecated!')
            df.to_csv(filename, index=False)
        print('saved celeba attributes to {}'.format(filename))
        return df

    def subset(self, idxs, overall_idx=False, complementary=False, max_attributes=9):
        if isinstance(idxs, int):
            selection = self.df.iloc[:,idxs]
        elif isinstance(idxs, list):
            selection = self.df.iloc[:, np.array(idxs)]
        else:
            raise TypeError(f'Expected arg. idxs of type: int or list, got {type(idxs)}')

        if complementary:
            if isinstance(idxs, int):
                attribute_names = [selection.name]
            elif isinstance(idxs, list):
                attribute_names = list(selection.columns)
            complement_categs_names = ['comp_' + c for c in attribute_names]
            # concatenate selected attributes df with its complementary.
            selection= pd.concat([selection, (self.df.iloc[:, np.array(idxs)] == 0)], axis=1)
            selection.columns = attribute_names + complement_categs_names
        if overall_idx:
            index_all = (selection== 1).any(axis=1)
            selection = pd.concat([selection, index_all], axis=1)
        return selection

    def pick_last_n_per_attribute(self, idxs, n=1):
        # select one or more attribute,
        df1 = self.subset(idxs)
        attr_last_idcs = dict()
        prev_idcs = set()
        prev_idxs_list = list()
        if not isinstance(df1, pd.DataFrame):
            df1 = pd.DataFrame(df1)

        for col in df1.columns:
            count = n
            # column_idxs = df1.iloc[::-1, c_i].nlargest(n)
            column_idxs = list(df1[col][::-1].nlargest(count).index)

            while prev_idcs.intersection(set(column_idxs)):
                # while any(idx in column_idxs for prev_idx in attr_last_idcs):
                count += 1
                column_idxs = list(df1[col][::-1].nlargest(count).index)
                column_idxs = column_idxs[count-n:]

            attr_last_idcs[col] = column_idxs
            prev_idcs = prev_idcs | set(column_idxs)
            prev_idxs_list.extend(column_idxs) # lists preserve order.

        df = pd.DataFrame(attr_last_idcs)
        df.columns = df1.columns
        return df, prev_idxs_list


    def categorise(self, attribute_subset):

        if len(attribute_subset.shape) > 1:
            out_arr = attribute_subset.iloc[:, 0].copy()
            n_attributes = attribute_subset.shape[1]
        if len(attribute_subset.shape) == 1:
            out_arr = attribute_subset.copy()
            attribute_subset = pd.DataFrame(attribute_subset)
            n_attributes = 1

        from itertools import product
        
        category_map = dict()
        for cat, bools in enumerate(product([0, 1], repeat=n_attributes)):
            # cat: column_n in attribute_subset.
            # bools
            boolean_intersection = []
            category = ''

            for i in range(n_attributes):
                # Cartes. prod. is going to provide, at each iteration, 
                # a list of booleans as long as the n_attributes.
                # We check the truth value of each category,
                # prefix the column name correspondingly.
                if not bools[i]:
                    category += 'comp-'
                category += attribute_subset.columns[i]
                boolean_intersection.append((attribute_subset.iloc[:, i] == bools[i]))
                if i != n_attributes-1:
                    category += '/'

            bool_inter = pd.concat(boolean_intersection, axis=1).all(axis=1)
            out_arr[bool_inter] = cat
            category_map[category] = cat
        return (out_arr, category_map)

    def categorise_subset(self, idxs, onehot_subset=False):

        attribute_subset = self.subset(idxs)
        if isinstance(idxs, int):
            return (attribute_subset, {f'comp-{attribute_subset.name}': 0,
                                       attribute_subset.name: 1}
                   )
        elif isinstance(idxs, list):
            if len(idxs) == 1:
                return (attribute_subset, {f'comp-{attribute_subset.name}': 0,
                                       attribute_subset.name: 1}
                      )
            elif len(idxs) > 1:
                categories_array = self.categorise(attribute_subset)
                return categories_array


def select_model(model_root, analyse_version=None, vmarker_fn='/version', 
                     epoch_range=(430, 690), test_epoch=False, granularity=10, figures=6):
    ''' Select EPOCH according to version specifications. (change function name)
    Out: 
        - fp_vmarker : int --  file containing `version`
        - fp_model : str -- file to model.pth.tar
        '''
    verified_ver = True
    while verified_ver:
        if test_epoch:
            assert isinstance(test_epoch, int), 'Epoch must be int'
            model_epoch = test_epoch
        else:
            model_epoch = randrange(epoch_range[0], epoch_range[1], granularity) # TODO only produce n%10==0
        fp_model_root = model_root + f'/epoch_{str(model_epoch).zfill(figures)}' #  + str(model_epoch)
        fp_vmarker = fp_model_root + vmarker_fn
        if test_epoch:
            break
        else:
            verified_ver = verify_version(fp_vmarker, str(analyse_version))
    fp_model = fp_model_root + '/model.pth.tar'
    print('selected model at {}th epoch'.format(model_epoch))
    return fp_model_root, fp_model, fp_vmarker

if __name__=='__main__':

    # RRRC = RandomRotatedResizedCrop(output_size = 256)

    # dataset = iter(FFHQLandmarks('ffhq-dataset-v2.json', 'data/FFHQ' , transform = RRRC))
    dataloader = iter(sample_FFHQ_eyes(1, 256, shuffle=False,
                                       transform=RandomRotatedResizedCrop(output_size=256)))

    viewpoint = 0

    its = int(70000/16)


    for i in range(its):
        sample, y = next(dataloader)
        # print('Img. filepaths:', fps)
        # import sys
        # print("What went wrong: ", sys.exc_info()[0])
        # print("And all just because:", sys.exc_info()[1])
        # print("Traceback:", sys.exc_info()[2])
        # print(i, ':', sample.shape, y)

        ax = plt.subplot(3, 4, i-viewpoint+1)
        # plt.tight_layout()
        ax.set_title(f'Sample #{i}')
        ax.axis('off')
        show_landmarks(image=sample, landmarks=y)
        if i == viewpoint+11:
            plt.show()
            break
