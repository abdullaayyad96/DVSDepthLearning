import numpy as np
import sys
import h5py
from random import shuffle


class DataLoader:
    def __init__(self, filename):
        self.data_ = h5py.File(filename)
        self.target_image_height_ = 224
        self.target_image_width_ = 320

    def get_batches_fn(self, batch_size):
        # shuffle data
        ind_list = [i for i in range(len(self.data_['event_images']))]
        shuffle(ind_list)
        event_images = [self.data_['event_binned_images'][i] for i in ind_list]
        edroded_depth_images = [self.data_['eroded_depth_images'][i] for i in ind_list]
        depth_edge_images = [self.data_['event_masked_depth_images'][i] for i in ind_list]
        diluted_event_mask_images = [self.data_['diluted_event_mask_images'][i] for i in ind_list]

        for batch_i in range(0, len(event_images), batch_size):
            inputs = []
            outputs = []
            weights = []
            weights2 = []
            for i in range(batch_i, np.min([batch_i+batch_size, len(event_images)])):
                temp_event_image = np.copy(event_images[i][0:self.target_image_height_, 0:self.target_image_width_, :])
                inputs.append(temp_event_image)
                output_image = depth_edge_images[i][0:self.target_image_height_, 0:self.target_image_width_].reshape((self.target_image_height_, self.target_image_width_, 1))
                output_image[np.isnan(output_image)] = 0.0
                outputs.append(output_image)

                cost_weights = edroded_depth_images[i][0:self.target_image_height_, 0:self.target_image_width_].reshape((self.target_image_height_, self.target_image_width_, 1))
                cost_weights[np.isnan(cost_weights)] = 0.0
                cost_weights[cost_weights>0] = 1.0
                weights.append(cost_weights)

                cost_weights2 = diluted_event_mask_images[i][0:self.target_image_height_, 0:self.target_image_width_].reshape((self.target_image_height_, self.target_image_width_, 1))
                cost_weights2[np.isnan(cost_weights2)] = 0.0
                cost_weights2[cost_weights2>0] = 1.0
                weights2.append(cost_weights2)


            yield np.array(inputs), np.array(outputs), np.array(weights), np.array(weights2)

    def load_all(self):

        event_images = self.data_['event_binned_images']
        edroded_depth_images = self.data_['eroded_depth_images']
        depth_edge_images = self.data_['event_masked_depth_images']
        diluted_event_mask_images = self.data_['diluted_event_mask_images']

        return event_images, depth_edge_images, edroded_depth_images, diluted_event_mask_images

