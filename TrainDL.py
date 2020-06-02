import tensorflow.compat.v1 as tf
import tensorflow as tf2
import numpy as np 
from DataLoader import DataLoader

class TrainDL:

    def train_nn(sess, epochs, batch_size, data_loader, train_op, loss_function, input_tensor,
             truth_image, image_mask_1, image_mask_2, learning_rate, base_learning_rate,
             learning_decay_rate, learning_decay_factor):
        """
        Train neural network and print    out the loss during training.
    param sess: TF Session
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param data_loader: Object of DataLoader type. Call using get_batches_fn(batch_size)
        :param train_op: TF Operation to train the neural network
        :param loss_function: TF Tensor for the amount of loss
        :param input_tensor: TF Placeholder for input images
        :param truth_image: TF Placeholder for truth images
        :param image_mask_1 and image_mask_2: TF Placeholders for masking NN output
        :param learning_rate: TF Placeholder for learning rate
        :param base_learning_rate: Float for the base learning rate of optimizer
        :param learning_decay_rate: Float for the period of dropping the learning rate
        :param learning_decay_factor: Float for decaying learning rate
        """
        #initialize variables
        sess.run(tf.global_variables_initializer())
        
        print("Training...")
        print()
        scaling_rate = 1
        
        loss_output = 0
        for i in range(epochs):
            loss_output = 0
            N_training_points = 0
            print("EPOCH {} ...".format(i+1))
            if i%learning_decay_rate == 0 and i != 0:
                scaling_rate = learning_decay_factor * scaling_rate

            for image, depth_image, mask_1, mask_2 in data_loader.get_batches_fn_timeseries(batch_size):

                optimizer, loss = sess.run([train_op, loss_function], 
                                feed_dict={input_tensor: image, truth_image: depth_image, image_mask_1: mask_1, image_mask_2: mask_2, learning_rate: scaling_rate*base_learning_rate})
                loss_output = loss_output + loss * len(image)
                N_training_points = N_training_points + len(image)
            
            print("Loss: =")
            print(loss_output / N_training_points)     
            print()

    def MSEOptimize(nn_last_layer, correct_value, image_mask_1, image_mask_2, learning_rate):
        """
        Build the TensorFLow loss and optimizer operations.
        :param nn_last_layer: TF Tensor of the last layer in the neural network
        :param correct_label: TF Placeholder for the truth images       
        :param image_mask_1 and image_mask_2: TF Placeholders for masking NN output
        :param learning_rate: TF Placeholder for the learning rate of optimizer
        :return: Tuple of (logits, train_op, cross_entropy_loss)
        """    
        weight_matrix = tf.math.multiply(image_mask_1, image_mask_2)
        mse_loss = tf.math.reduce_sum(tf.math.square( tf.math.subtract(tf.math.multiply(correct_value, weight_matrix), tf.math.multiply(nn_last_layer, weight_matrix))))

        normalized_loss = tf2.math.divide(mse_loss,  tf2.cast(tf.math.count_nonzero(weight_matrix), dtype=tf.float32))

        #obtain training operation
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-8) #Note default value of epsilon 1e-8 results in instability after few epochs
    
        #clip the gradients
        gvs = optimizer.compute_gradients(normalized_loss)
        #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        training_operation = optimizer.apply_gradients(gvs)

        return training_operation, normalized_loss

    
    def MSNEOptimize(nn_last_layer, correct_value, image_mask_1, image_mask_2, learning_rate):
        """
        Build the TensorFLow loss and optimizer operations.
        :param nn_last_layer: TF Tensor of the last layer in the neural network
        :param correct_label: TF Placeholder for the truth images       
        :param image_mask_1 and image_mask_2: TF Placeholders for masking NN output
        :param learning_rate: TF Placeholder for the learning rate of optimizer
        :return: Tuple of (logits, train_op, cross_entropy_loss)
        """    
        weight_matrix = tf.math.multiply(image_mask_1, image_mask_2)
        error = tf.math.subtract(tf.math.multiply(correct_value, weight_matrix), tf.math.multiply(nn_last_layer, weight_matrix))
        normalized_error = tf.math.divide_no_nan(error, tf.math.multiply(correct_value, weight_matrix))
        mse_loss = tf.math.reduce_sum(tf.math.square(normalized_error))

        normalized_loss = tf2.math.divide(mse_loss,  tf2.cast(tf.math.count_nonzero(weight_matrix), dtype=tf.float32))
        
        #obtain training operation
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-8) #Note default value of epsilon 1e-8 results in instability after few epochs
    
        #clip the gradients
        gvs = optimizer.compute_gradients(normalized_loss)
        #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        training_operation = optimizer.apply_gradients(gvs)

        return training_operation, normalized_loss

    
    def HuberOptimize(nn_last_layer, correct_value, image_mask_1, image_mask_2, learning_rate):
        """
        Build the TensorFLow loss and optimizer operations.
        :param nn_last_layer: TF Tensor of the last layer in the neural network
        :param correct_label: TF Placeholder for the truth images       
        :param image_mask_1 and image_mask_2: TF Placeholders for masking NN output
        :param learning_rate: TF Placeholder for the learning rate of optimizer
        :return: Tuple of (logits, train_op, cross_entropy_loss)
        """    
        weight_matrix = tf.math.multiply(image_mask_1, image_mask_2)
        y_pred =  tf.math.multiply(nn_last_layer, weight_matrix)    
        y_truth = tf.math.multiply(correct_value, weight_matrix)
        huber_loss = tf2.keras.losses.Huber()(y_truth, y_pred)

        #normalized_loss = tf2.math.divide(huber_loss,  tf2.cast(tf.math.count_nonzero(weight_matrix), dtype=tf.float32))
        
        #obtain training operation
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-8) #Note default value of epsilon 1e-8 results in instability after few epochs
    
        #clip the gradients
        gvs = optimizer.compute_gradients(huber_loss)
        #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        training_operation = optimizer.apply_gradients(gvs)

        return training_operation, huber_loss


    def NormalizedHuberOptimize(nn_last_layer, correct_value, image_mask_1, image_mask_2, learning_rate):
        """
        Build the TensorFLow loss and optimizer operations.
        :param nn_last_layer: TF Tensor of the last layer in the neural network
        :param correct_label: TF Placeholder for the truth images       
        :param image_mask_1 and image_mask_2: TF Placeholders for masking NN output
        :param learning_rate: TF Placeholder for the learning rate of optimizer
        :return: Tuple of (logits, train_op, cross_entropy_loss)
        """    
        weight_matrix = tf.math.multiply(image_mask_1, image_mask_2)
        y_pred =  tf2.boolean_mask(nn_last_layer, weight_matrix)    
        y_truth = tf2.boolean_mask(correct_value, weight_matrix)
        huber_loss = tf2.keras.losses.Huber()(y_truth, y_pred)

        #normalized_loss = tf2.math.divide(huber_loss,  tf2.cast(tf.math.count_nonzero(weight_matrix), dtype=tf.float32))
        
        #obtain training operation
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-8) #Note default value of epsilon 1e-8 results in instability after few epochs
    
        #clip the gradients
        gvs = optimizer.compute_gradients(huber_loss)
        #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        training_operation = optimizer.apply_gradients(gvs)

        return training_operation, huber_loss