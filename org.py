import os
import shutil

# Define the mapping of files to folders
file_mapping = {
    "Preprocessing": [
        "data_preprocessing_tools.ipynb",
        "feature_reduction.ipynb",
        "pandas.ipynb",
        "visualization.ipynb"
    ],
    "Regression": [
        "linear_regression_with_synthetic_data.ipynb",
        "linear_regression_with_a_real_dataset.ipynb",
        "simple_linear_regression.ipynb",
        "multiple_linear_regression.ipynb",
        "polynomial_regression.ipynb",
        "support_vector_regression.ipynb",
        "decision_tree_regression.ipynb",
        "random_forest_regression.ipynb"
    ],
    "Classification": [
        "logistic_regression.ipynb",
        "naive_bayes.ipynb",
        "decision_tree_classification.ipynb",
        "random_forest_classification.ipynb",
        "k_nearest_neighbors.ipynb",
        "support_vector_machine.ipynb",
        "kernel_svm.ipynb",
        "kernel_pca.ipynb",
        "linear_discriminant_analysis.ipynb",
        "hierarchical_clustering.ipynb",
        "k_means_clustering.ipynb",
        "mnist_classification.ipynb",
        "imageclassification.ipynb",
        "xg_boost.ipynb"
    ],
    "NLP": [
        "natural_language_processing.ipynb",
        "NLP.ipynb"
    ],
    "Reinforcement_Learning": [
        "thompson_sampling.ipynb",
        "upper_confidence_bound.ipynb"
    ],
    "Generative_Models": [
        "GAN.ipynb"
    ]
}

# Current directory (you can change this if needed)
base_dir = "."

# Create folders and move files
for folder, files in file_mapping.items():
    folder_path = os.path.join(base_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
    for file in files:
        src_path = os.path.join(base_dir, file)
        dest_path = os.path.join(folder_path, file)
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
            print(f"Moved {file} --> {folder}")
        else:
            print(f"File not found: {file}")

print("\n✅ All files organized successfully!")
