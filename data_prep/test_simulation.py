from offline_input_data_gen import *
import scipy.io as sio
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
		
if __name__ == "__main__":

	# Data settings
	parser = argparse.ArgumentParser(description='Test simulation')
	parser.add_argument('--mask-path', type=str, default='../experiment_mask.mat', metavar='Path', help='Mask .mat file')
	args = parser.parse_args()

	input_scene = sio.loadmat('test_scene.mat')
	input_scene = input_scene[sorted(input_scene.keys())[-1]]

	test_streaking_image = transforms.ToTensor()(Image.open('test_simulation.png')).numpy().squeeze(0)

	mask = sio.loadmat(args.mask_path)
	mask = mask[sorted(mask.keys())[-1]]

	output_streaking_image = get_streaking_image(x=input_scene, mask=mask, intensity_variation=False)

	print(np.allclose(output_streaking_image, test_streaking_image, rtol=1e-5, atol=1e-8))

	diff = np.abs( output_streaking_image - test_streaking_image)

	print(diff.max(), diff.min(), diff.mean(), diff.std(), np.ones_like(diff)[np.where(diff>1e-2)].sum()/diff.size)