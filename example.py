from src.pan_ai_x.pan_ai_x import PAN_AI_X

# Example usage:
weights_path = '/.../model.pth'
image_path = '/.../image.png'
train_dataset_path = '/.../train_dir'
val_dataset_path = '/.../val_dir'
test_dataset_path = '/.../test_dir'

# Create an instance of ToothPredictor
tooth_predictor = PAN_AI_X(weights_path)

# Use predict method for inference
prediction = tooth_predictor.predict(image_path)
print(prediction)

# Use train method for training
tooth_predictor.train(train_dataset_path, val_dataset_path, test_dataset_path)