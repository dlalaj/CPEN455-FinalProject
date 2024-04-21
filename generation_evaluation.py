'''
This code is used to evaluate the FID score of the generated images.
You should at least guarantee this code can run without any error on test set.
And whether this code can run is the most important factor for grading.
We provide the remaining code,  you can't modify the remaining code, all you should do are:
1. Modify the sample function to get the generated images from the model and ensure the generated images are saved to the gen_data_dir(line 12-18)
2. Modify how you call your sample function(line 31)
'''
from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *
from model import * 
from dataset import *
import os
import torch
# You should modify this sample function to get the generated images from your model
# This function should save the generated images to the gen_data_dir, which is fixed as 'samples'
# Begin of your code
sample_op = lambda x : sample_from_discretized_mix_logistic(x, 10)
# NOTE: Here without setting sample_batch_size to 48 I am unable to compute the FID score, please take this
# into account. It was an issue brought up on Piazza a lot and as far as I could tell no proper solution for
# it was provided
def my_sample(model, gen_data_dir, sample_batch_size = 48, obs = (3,32,32), sample_op = sample_op):
    # Iterate over the values do not care about strings but their numerical classes instead
    for label in my_bidict.values():
        print(f"Label: {label}")
        # generate images for each label, each label has 25 images
        
        # Prepare label tensor and move it to the correct device, see example in sample() function for this
        img_labels = torch.full((sample_batch_size,), label)
        img_labels = img_labels.to(next(model.parameters()).device)

        sample_t = sample(model, sample_batch_size, obs, sample_op, img_labels)
        sample_t = rescaling_inv(sample_t)
        save_images(sample_t, os.path.join(gen_data_dir), label=label)
    pass
# End of your code

if __name__ == "__main__":
    ref_data_dir = "data/test"
    gen_data_dir = "samples"
    BATCH_SIZE=128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)
    #Begin of your code
    #Load your model and generate images in the gen_data_dir
    model = PixelCNN(nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=10)
    # TODO: Need to load pre-trained model here
    model = model.to(device)
    model = model.eval()
    # Load model, see example in classification_evaluation.py provided by TA
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth'))
    my_sample(model=model, gen_data_dir=gen_data_dir)
    # End of your code
    paths = [gen_data_dir, ref_data_dir]
    print("#generated images: {:d}, #reference images: {:d}".format(
        len(os.listdir(gen_data_dir)), len(os.listdir(ref_data_dir))))

    try:
        fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
        print("Dimension {:d} works! fid score: {}".format(192, fid_score, gen_data_dir))
    except:
        print("Dimension {:d} fails!".format(192))
        
    print("Average fid score: {}".format(fid_score))
