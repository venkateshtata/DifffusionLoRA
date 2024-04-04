from datasets import load_dataset
import os

# Load the dataset (replace 'dataset_name' with the actual name of the dataset)
dataset = load_dataset("lambdalabs/pokemon-blip-captions")

# Specify the directory where you want to save the files
save_dir = './'
os.makedirs(save_dir, exist_ok=True)

# Iterate over the dataset and save each example
for split in dataset.keys():
    split_dir = os.path.join(save_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    # Replace 'text_column_name' with the actual name of the column you want to save
    for i, example in enumerate(dataset[split]):
        file_path = os.path.join(split_dir, f'{i}.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(example['text_column_name'])
