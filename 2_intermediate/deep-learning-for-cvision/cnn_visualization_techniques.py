import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import warnings
warnings.filterwarnings('ignore')

class CNNVisualization:
    """Class for visualizing CNN internals and understanding"""
    
    def __init__(self):
        self.model = None
        
    def load_pretrained_model(self):
        """Load a pre-trained model for visualization"""
        model = applications.VGG16(weights='imagenet', include_top=True)
        return model
    
    def visualize_feature_maps(self, model, image_path, layer_names=None):
        """Visualize feature maps from different layers"""
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = applications.vgg16.preprocess_input(img_array)
        
        if layer_names is None:
            layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        
        # Create models that output the feature maps
        outputs = [model.get_layer(name).output for name in layer_names]
        feature_map_model = models.Model(inputs=model.input, outputs=outputs)
        
        # Get feature maps
        feature_maps = feature_map_model.predict(img_array)
        
        # Plot original image and feature maps
        plt.figure(figsize=(20, 15))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Feature maps for each layer
        for i, (feature_map, layer_name) in enumerate(zip(feature_maps, layer_names)):
            # Plot first few feature maps
            plt.subplot(2, 3, i + 2)
            # Use the first feature map for visualization
            plt.imshow(feature_map[0, :, :, 0], cmap='viridis')
            plt.title(f'{layer_name}\nFeature Map 1')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return feature_maps
    
    def visualize_filters(self, model, layer_name='block1_conv1'):
        """Visualize convolutional filters"""
        layer = model.get_layer(layer_name)
        filters, biases = layer.get_weights()
        
        # Normalize filter values to 0-1 for visualization
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        
        # Plot first few filters
        n_filters = min(16, filters.shape[3])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        
        for i in range(n_filters):
            ax = axes[i//4, i%4]
            # Get the filter
            f = filters[:, :, :, i]
            # For RGB filters, convert to grayscale or show channels separately
            if f.shape[2] == 3:
                # Convert to grayscale for visualization
                f_gray = np.mean(f, axis=2)
                ax.imshow(f_gray, cmap='gray')
            else:
                ax.imshow(f[:, :, 0], cmap='gray')
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
        
        plt.suptitle(f'Convolutional Filters - {layer_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return filters
    
    def create_class_activation_map(self, model, image_path, class_idx=None):
        """Create Class Activation Map (CAM) for a specific class"""
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = applications.vgg16.preprocess_input(img_array)
        
        # Get the last convolutional layer and the predictions
        last_conv_layer = model.get_layer('block5_conv3')
        classifier_layers = [layer for layer in model.layers if layer.name != 'block5_conv3']
        
        # Create a model that outputs the last conv layer and the predictions
        cam_model = models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )
        
        # Get the conv output and predictions
        conv_output, predictions = cam_model.predict(img_array)
        conv_output = conv_output[0]  # Remove batch dimension
        
        # If no class specified, use the predicted class
        if class_idx is None:
            class_idx = np.argmax(predictions[0])
        
        # Get the weights for the class from the last dense layer
        class_weights = model.get_layer('predictions').get_weights()[0]
        class_weights = class_weights[:, class_idx]
        
        # Create the class activation map
        cam = np.zeros(conv_output.shape[0:2])
        for i, w in enumerate(class_weights):
            cam += w * conv_output[:, :, i]
        
        # ReLU activation
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam / cam.max()
        
        # Resize to original image size
        cam = cv2.resize(cam, (224, 224))
        
        # Convert to heatmap
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose on original image
        img_original = cv2.imread(image_path)
        img_original = cv2.resize(img_original, (224, 224))
        superimposed_img = heatmap * 0.4 + img_original
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Class Activation Map')
        axes[1].axis('off')
        
        # Superimposed
        axes[2].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Superimposed')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print prediction
        class_names = applications.imagenet_utils.decode_predictions(predictions)[0]
        print("Top predictions:")
        for i, (imagenet_id, label, score) in enumerate(class_names):
            print(f"{i+1}: {label} ({score:.2f})")
        
        return cam, superimposed_img
    
    def visualize_layer_outputs(self, model, image_path, layer_names):
        """Visualize outputs of specific layers"""
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = applications.vgg16.preprocess_input(img_array)
        
        # Create models for each layer
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        for i, layer_name in enumerate(layer_names[:5]):
            layer = model.get_layer(layer_name)
            layer_model = models.Model(inputs=model.input, outputs=layer.output)
            layer_output = layer_model.predict(img_array)
            
            # Visualize first few channels
            n_channels = min(3, layer_output.shape[-1])
            for j in range(n_channels):
                axes[i+1].imshow(layer_output[0, :, :, j], cmap='viridis')
            axes[i+1].set_title(f'{layer_name}\nOutput')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_convolution_operation(self):
        """Demonstrate how convolution operation works"""
        # Create a simple image with patterns
        image = np.zeros((50, 50))
        image[10:20, 10:20] = 1  # Square
        image[30:40, 30:40] = 1  # Another square
        
        # Create different filters
        filters = {
            'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            'Edge Detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            'Blur': np.ones((3, 3)) / 9
        }
        
        # Apply convolution
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        
        for i, (filter_name, kernel) in enumerate(filters.items()):
            # Apply convolution
            filtered = cv2.filter2D(image, -1, kernel)
            
            # Plot original and filtered
            axes[0, i].imshow(image, cmap='gray')
            axes[0, i].set_title('Original Image')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(filtered, cmap='gray')
            axes[1, i].set_title(f'{filter_name} Filter')
            axes[1, i].axis('off')
        
        plt.suptitle('Convolution Operation Demonstration', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def visualize_pooling_operations(self):
        """Demonstrate different pooling operations"""
        # Create a sample feature map
        feature_map = np.random.rand(8, 8) * 10
        
        # Apply different pooling operations
        max_pooled = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(feature_map[np.newaxis, :, :, np.newaxis])
        avg_pooled = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(feature_map[np.newaxis, :, :, np.newaxis])
        
        # Convert back to numpy
        max_pooled = max_pooled.numpy().squeeze()
        avg_pooled = avg_pooled.numpy().squeeze()
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(feature_map, cmap='viridis')
        axes[0].set_title('Original Feature Map')
        axes[0].axis('off')
        
        axes[1].imshow(max_pooled, cmap='viridis')
        axes[1].set_title('Max Pooling (2x2)')
        axes[1].axis('off')
        
        axes[2].imshow(avg_pooled, cmap='viridis')
        axes[2].set_title('Average Pooling (2x2)')
        axes[2].axis('off')
        
        # Add values to plots
        for i in range(feature_map.shape[0]):
            for j in range(feature_map.shape[1]):
                axes[0].text(j, i, f'{feature_map[i, j]:.1f}', 
                           ha='center', va='center', color='white', fontsize=8)
        
        for i in range(max_pooled.shape[0]):
            for j in range(max_pooled.shape[1]):
                axes[1].text(j, i, f'{max_pooled[i, j]:.1f}', 
                           ha='center', va='center', color='white', fontsize=10)
                axes[2].text(j, i, f'{avg_pooled[i, j]:.1f}', 
                           ha='center', va='center', color='white', fontsize=10)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run CNN visualization techniques"""
    print("=== CNN Visualization and Understanding ===")
    
    # Initialize visualization class
    viz = CNNVisualization()
    
    # Load pre-trained model
    model = viz.load_pretrained_model()
    print("Loaded VGG16 model for visualization")
    
    # You'll need an actual image file for these visualizations
    # For demonstration, we'll use a sample image path
    # Replace with actual image path for real usage
    sample_image_path = "sample_image.jpg"  # Replace with actual path
    
    print("\n1. Demonstrating Convolution Operation...")
    viz.demonstrate_convolution_operation()
    
    print("\n2. Demonstrating Pooling Operations...")
    viz.visualize_pooling_operations()
    
    print("\n3. Visualizing Convolutional Filters...")
    filters = viz.visualize_filters(model, 'block1_conv1')
    
    # Note: The following require actual image files
    print("\nNote: Feature map and CAM visualizations require actual image files.")
    print("Please provide image paths to see those visualizations.")
    
    # Example of how to use with actual images:
    """
    print("\n4. Visualizing Feature Maps...")
    feature_maps = viz.visualize_feature_maps(model, 'path/to/your/image.jpg')
    
    print("\n5. Creating Class Activation Map...")
    cam, superimposed = viz.create_class_activation_map(model, 'path/to/your/image.jpg')
    """
    
    return viz, model

if __name__ == "__main__":
    # Run CNN visualization techniques
    viz_tool, pretrained_model = main()
    
    print("\n=== CNN Visualization Completed ===")
    print("Various CNN visualization techniques demonstrated successfully!")