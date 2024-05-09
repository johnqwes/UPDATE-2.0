from joblib import dump, load
import pickle

# Define the file paths for the pickle files and corresponding joblib files
pickle_files = ['stacked_model_drop.pkl', 'stacked_model_enroll.pkl', 'stacked_model_grad.pkl', 'stacked_model.pkl']
joblib_files = ['stacked_model_drop.joblib', 'stacked_model_enroll.joblib', 'stacked_model_grad.joblib', 'stacked_model.joblib']

# Convert each pickle file to joblib file
for pickle_file, joblib_file in zip(pickle_files, joblib_files):
    with open(pickle_file, 'rb') as f:
        model = pickle.load(f)
    dump(model, joblib_file)

# Check if the conversion is successful by loading the joblib files
for joblib_file in joblib_files:
    loaded_model = load(joblib_file)
    print(f"Model loaded from {joblib_file}: {loaded_model}")