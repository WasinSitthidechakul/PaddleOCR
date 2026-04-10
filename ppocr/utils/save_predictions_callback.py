"""
Callback to save prediction examples during training
Automatically generates prediction samples every N epochs
"""

import os
import cv2
import numpy as np
import paddle


class SavePredictionsCallback:
    """
    Saves prediction examples during training evaluation
    """

    def __init__(self, config, save_dir='output/thai_lpr_simple/prediction_examples', save_interval=20, num_samples=50):
        """
        Args:
            config: Training configuration
            save_dir: Directory to save predictions
            save_interval: Save predictions every N epochs
            num_samples: Number of samples to save per dataset
        """
        self.config = config
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.num_samples = num_samples

        os.makedirs(save_dir, exist_ok=True)

        # Load dataset paths
        self.datasets = {}
        if 'Train' in config and 'dataset' in config['Train']:
            train_labels = config['Train']['dataset'].get('label_file_list', [])
            if train_labels:
                self.datasets['train'] = train_labels[0]

        if 'Eval' in config and 'dataset' in config['Eval']:
            eval_labels = config['Eval']['dataset'].get('label_file_list', [])
            if eval_labels:
                self.datasets['val'] = eval_labels[0]

        # Try to find test dataset
        test_path = 'train_data/thai_lpr/test_list.txt'
        if os.path.exists(test_path):
            self.datasets['test'] = test_path

        self.data_dir = config['Train']['dataset'].get('data_dir', './')

    def save_predictions(self, model, post_process_class, epoch, character_dict):
        """
        Save predictions for all datasets
        """
        if epoch % self.save_interval != 0:
            return

        print(f"\n{'='*80}")
        print(f"Saving prediction examples at epoch {epoch}")
        print(f"{'='*80}\n")

        model.eval()

        for dataset_name, label_file in self.datasets.items():
            try:
                output_file = os.path.join(self.save_dir, f'{dataset_name}_predictions_epoch_{epoch}.txt')
                self._save_dataset_predictions(model, post_process_class, label_file,
                                               output_file, dataset_name, epoch, character_dict)
            except Exception as e:
                print(f"Error saving {dataset_name} predictions: {e}")

        print(f"{'='*80}")
        print(f"Prediction examples saved to: {self.save_dir}")
        print(f"{'='*80}\n")

        model.train()

    def _save_dataset_predictions(self, model, post_process_class, label_file,
                                  output_file, dataset_name, epoch, character_dict):
        """
        Save predictions for a single dataset
        """
        # Read labels
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Limit samples
        lines = lines[:self.num_samples]

        results = []
        correct_count = 0

        for idx, line in enumerate(lines):
            try:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue

                img_path = parts[0]
                ground_truth = parts[1]

                # Full path
                full_path = os.path.join(self.data_dir, img_path)

                if not os.path.exists(full_path):
                    continue

                # Read and preprocess image
                img = cv2.imread(full_path)
                if img is None:
                    continue

                # Resize to model input size
                img = cv2.resize(img, (100, 32))
                img = img.astype('float32')

                # Normalize
                img = img / 255.0
                img = (img - 0.5) / 0.5  # Normalize to [-1, 1]

                # Transpose to CHW
                img = img.transpose((2, 0, 1))
                img = img[np.newaxis, :]

                # Convert to tensor
                img_tensor = paddle.to_tensor(img, dtype='float32')

                # Predict
                with paddle.no_grad():
                    preds = model(img_tensor)

                # Post process
                post_result = post_process_class(preds)

                if isinstance(post_result, dict):
                    prediction = post_result.get('texts', [''])[0]
                    confidence = post_result.get('scores', [0.0])[0]
                elif isinstance(post_result, list) and len(post_result) > 0:
                    if isinstance(post_result[0], tuple):
                        prediction = post_result[0][0]
                        confidence = post_result[0][1]
                    else:
                        prediction = str(post_result[0])
                        confidence = 0.0
                else:
                    prediction = "ERROR"
                    confidence = 0.0

                # Check if correct
                is_correct = "✓" if prediction == ground_truth else "✗"
                if prediction == ground_truth:
                    correct_count += 1

                result_line = f"{is_correct} Ground truth = {ground_truth:15s} | Prediction = {prediction:15s} | Conf = {confidence:.4f}"
                results.append(result_line)

            except Exception as e:
                results.append(f"ERROR processing {img_path}: {e}")
                continue

        # Calculate accuracy
        accuracy = (correct_count / len(results) * 100) if results else 0.0

        # Write results
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*100}\n")
            f.write(f"{dataset_name.upper()} Dataset Predictions - Epoch {epoch}\n")
            f.write(f"{'='*100}\n")
            f.write(f"Total samples: {len(results)}\n")
            f.write(f"Correct predictions: {correct_count}/{len(results)}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n")
            f.write(f"{'='*100}\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"{i:3d}. {result}\n")

        print(f"  {dataset_name:10s}: {correct_count:3d}/{len(results):3d} correct ({accuracy:5.2f}%) -> {output_file}")


def create_prediction_callback(config):
    """
    Factory function to create callback
    """
    save_dir = os.path.join(config['Global']['save_model_dir'], 'prediction_examples')
    save_interval = config['Global'].get('save_epoch_step', 20)  # Save at same interval as model checkpoints

    return SavePredictionsCallback(config, save_dir, save_interval, num_samples=50)
