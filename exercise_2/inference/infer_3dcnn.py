"""Utility for inference using trained networks"""

import torch

from exercise_2.data.shapenet import ShapeNetVox
from exercise_2.model.cnn3d import ThreeDeeCNN


class InferenceHandler3DCNN:
    """Utility for inference using trained 3DCNN network"""

    def __init__(self, ckpt):
        """
        :param ckpt: checkpoint path to weights of the trained network
        """
        self.model = ThreeDeeCNN(ShapeNetVox.num_classes)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()

    def infer_single(self, voxels):
        """
        Infer class of the shape given its voxel grid representation
        :param voxels: voxel grid of shape 32x32x32
        :return: class category name for the voxels, as predicted by the model
        """
        input_tensor = torch.from_numpy(voxels).float().unsqueeze(0).unsqueeze(0)

        # TODO: Predict class
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            print('Using CPU')

        input_tensor = input_tensor.to(device)
        self.model.to(device)
        prediction = self.model(input_tensor)
        class_id = torch.argmax(prediction[:, 0, :], dim=1)
        class_name = ShapeNetVox.classes[class_id]
        # class id to class name
        class_name = ShapeNetVox.class_name_mapping[class_name]

        return class_name
