#!/var/scratch/mao540/miniconda3/envs/maip-venv/bin/python

from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF

from skimage import io
from PIL import Image, ImageDraw
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import random  


def sample_from_directory(path, batch, image_size, test=False):
    if not test:
        shuffle = True
        split = 'whole dataset'
        transform = transforms.Compose([
            # transforms.CenterCrop(160),
            transforms.Resize(size=image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])
    else:
        split = 'test'
        transform = transforms.Compose([
        # transforms.CenterCrop(160),
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])
    print(f'shuffle set to {shuffle} for split: {split}')
    # target_type = ['attr', 'bbox', 'landmarks']
    dataset = datasets.ImageFolder(root = path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=8)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=8)
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
    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0)
    loader = iter(loader)
    
    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = DataLoader(dataset, batch_size=batch, shuffle=True) # , num_workers=16)
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

    def __init__(self, output_size, crop_margin = 20, degrees=90):
        self.output_size = output_size # 256
        self.crop_margin = crop_margin
        self.to_tensor = transforms.ToTensor()
        self.circular_reps = (2, 4)
        self.concentric_reps = 2
        print(f'PARAMS: crop_margin: {crop_margin}, circular reps: {self.circular_reps}', end='')

    def __call__(self, input):

        bg = False
        input_image, input_eyes = input

        for cr in range(self.concentric_reps):
            # import ipdb; ipdb.set_trace()

            # copy and pad original image, take bbox coordinates
            if not bg:
                image, eyes, ratio, inner_h = self._preproc_resize(input_image.copy(),
                                                             input_eyes.copy())
                base_image, bbox_coords, eyes, in_deg = self._init_rotate(
                                                            image.copy(), eyes, ratio)
                base_image, mask, rot_params = self._rotate_and_composite_n(
                                                          base_image, bbox_coords, bg)
                # prev_rotation = (init_rot_params, rot_params)
            else:
                image, eyes, ratio = self._preproc_resize(input_image.copy(),
                                                          input_eyes.copy(), inner_h)
                base_image, bbox_coords, eyes = self._init_bg(image.copy(), eyes, ratio,
                                                              in_deg, rot_params)
                base_image, mask = self._rotate_and_composite_n(base_image, bbox_coords, bg, rot_params)

            if not bg:
                fg_img = base_image.copy()
                fg_mask = mask.copy()
                fg_eyes = eyes.copy()
            bg = True

        # crop background
        base_image = self.crop_image(base_image)

        base_image = Image.composite(fg_img, base_image, fg_mask)

        print(base_image.size)
        image = self.to_tensor(base_image) # mask.convert('RGB')) 
        return image, fg_eyes
    
    def bbox_measures(self, eyes, side):
        min_x, min_y, max_x, max_y = self.take_bbox(eyes, self.crop_margin)
        eyes_crop_h = max_y - min_y
        eyes_crop_w = max_x - min_x
        diag = side * sqrt(2)

        ## Resize by giving crop height.
        # max_h: H for crop with W spanning 256 px
        max_h = int(self.output_size * (eyes_crop_h / eyes_crop_w))
        # diag_h: H for crop with W spanning `diag` pixels (~ 1448.15).
        diag_h = int(diag * (eyes_crop_h / eyes_crop_w))

        max_inner = int(self.crop_margin + max_h / 2)
        max_outer = int(self.crop_margin + diag_h / 2)
        crop_input = (min_y, min_x, eyes_crop_h, eyes_crop_w)
        return (max_inner, max_outer, crop_input)

    def _preproc_resize(self, image, eyes, inner_h=None):
        ''' 
        Crops area around eyes; resizes with smaller in range (self.crop_margin, max_h).
        Args:
            input (tuple): (PIL.Image, np.ndarray)
        '''

        max_inner, max_outer, crop_input = self.bbox_measures(eyes, image.height)

        if not inner_h: 
            H = random.randint(self.crop_margin, max_inner) # + self.crop_margin)
            print('inner range: ', self.crop_margin, max_inner, f'selected: {H}', end='')
        else:
            # = random.randint(max_inner, max_outer) # + self.crop_margin)
            H = inner_h * 2 # + 20
            print(' | outer range: ', max_inner, max_outer, f'selected: {H}', end='')

        image = TF.resized_crop(image.copy(), *crop_input, H) # , min_y, min_x, eyes_crop_h, eyes_crop_w, random_h)
        # update landmarks coordinates against
        new_w, new_h = image.size       # original size:
        eyes[:,:,0] -= crop_input[1]    # <- min_x
        eyes[:,:,1] -= crop_input[0]    # <- min_y
        ratio = new_h / crop_input[2]   # <- eyes_crop_h
        eyes *= ratio # and resize 
        if not inner_h:
            return (image, eyes, ratio, H)
        else: return (image, eyes, ratio)

    def _init_bg(self, image, eyes, ratio, in_degrees, rot_params):
        # square padding
        s = max(image.size)
        image, eyes = self.pad_image(image, size=(s, s), eyes=eyes, padding_mode='constant')
        # print('pre-rot.: ', image.size, ', post-rot.: ', end='')

        bbox_coords = self.take_bbox(eyes,
                           crop_margin = self.crop_margin * ratio)

        # random initial rotation.
        # _, in_degrees, in_radians = self.random_degrees(90)
        in_degrees = in_degrees - (rot_params[1] / 2)
        radians = in_degrees * 3.1415926 / 180
        image = image.rotate(- in_degrees, expand=False)

        # import ipdb; ipdb.set_trace()
        initRotation = RotationCoords(radians,
                               xy_t=np.array(image.size)/2, y_span=s)
        eyes = initRotation(np.array(eyes))
        bbox_vertices = initRotation(np.array(bbox_coords))

        # bbox_vertices = bbox_coords
        return (image, bbox_vertices, eyes)

    def _adjust_silhuette(self, bbox_vrtc, eyes):
        left_vtx = (bbox_vrtc[0,0], eyes[0,0,1])
        right_vtx = (bbox_vrtc[1, 0], eyes[1, 3, 1])
        insert_vrtc = np.array([left_vtx, right_vtx])

        bbox_vrtc[0, 0] = eyes[0, -1,0].copy() # bottom-left eyelid
        bbox_vrtc[1, 0] = eyes[1, -2,0].copy() # bottom-right eyelid
        bbox_vrtc[2, 0] = eyes[1, 2, 0].copy()
        bbox_vrtc[3, 0] = eyes[0, 1, 0].copy()
        # import ipdb; ipdb.set_trace()
        bbox_vrtc = np.insert(bbox_vrtc, (0, 2), insert_vrtc, axis=0)
        return bbox_vrtc

    def _init_rotate(self, image, eyes, ratio):
        ''' set initial random rotation on image and coordinates.'''
        # pad image, take bbox according to new ratio
        image, eyes = self.pad_image(image, eyes=eyes, padding_mode='constant')

        bbox_coords = self.take_bbox(eyes, 
                           crop_margin = self.crop_margin * ratio)
        # add corners
        # import ipdb; ipdb.set_trace()
        # XXX NEW
        bbox_vertices = _find_coords_vertices(*bbox_coords)
        bbox_vertices = self._adjust_silhuette(bbox_vertices, eyes)

        # random initial rotation. 
        _, in_degrees, in_radians = self.random_degrees(90)
        print('  Degrees', in_degrees)
        image = image.rotate(- in_degrees) # , expand=True)

        initRotation = RotationCoords(in_radians)
        eyes = initRotation(np.array(eyes))
        bbox_vertices = initRotation(np.array(bbox_vertices))
        # TODO bbox_vertices.jjkk

        return (image, bbox_vertices, eyes, in_degrees)

    def _rotate_and_composite_n(self, base_image, bbox_coords, bg=False,
                                  rotation_params = None):

        # TODO: compute random reps from range based on ratio
        if not rotation_params:
            reps, degrees, radians = self.random_degrees(*self.circular_reps)
        else:
            reps, degrees, radians = rotation_params

        maskRotation = RotationCoords(radians, xy_t=np.array(base_image.size)/2,
                                        y_span = base_image.height, verbose=False)

        mask = Image.new('1', base_image.size)
        draw_mask = ImageDraw.Draw(mask)
        
        draw_mask.polygon([tuple(c) for c in bbox_coords], fill=1 , outline=0)

        for r in range(reps):
            r_image = base_image.copy().rotate(- degrees * (r + 1), expand=False)
            # if not bg:
                # r_image = self.pad_image(r_image) # , size = (s,s))
            
            base_image = Image.composite(base_image, r_image, mask)

            if r != reps-1 or not bg:
                bbox_coords = maskRotation(coords=np.array(bbox_coords))
                draw_mask.polygon([tuple(c) for c in bbox_coords], fill=1, outline=0)

        if not bg:
            return (base_image, mask, (reps, degrees, radians))
        else:
            return (base_image, mask) # , degrees)
        

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


    def pad_image(self, image, size=None, padding_mode='constant', eyes = None):
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
        image = TF.pad(image, padding, 0 ,padding_mode)

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


if __name__=='__main__':

    # RRRC = RandomRotatedResizedCrop(output_size = 256)

    # dataset = iter(FFHQLandmarks('ffhq-dataset-v2.json', 'data/FFHQ' , transform = RRRC))
    dataloader = iter(sample_FFHQ_eyes(1, 256, shuffle=False, transform=RandomRotatedResizedCrop(output_size=256)))

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
