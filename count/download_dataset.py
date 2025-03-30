from datasets import load_dataset

def main():

    # Specify the directory where you want to save the dataset
    save_dir = "./count/countbenchqa_data"

    # Download and load the dataset
    dataset = load_dataset("vikhyatk/CountBenchQA", cache_dir=save_dir)

    # Optional: Print basic info about the dataset
    print(f"Dataset downloaded to: {save_dir}")
    print(f"Dataset structure: {dataset}")

if __name__ == '__main__':
    main()
