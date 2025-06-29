import subprocess

# Define the names of the Python files to run
file1 = 'train_discriminator.py'
file2 = 'train.py'

# Run the first Python file
for i in range(200):
    print(f"======================Iteration {i}=======================")
    subprocess.run(['python', file1])

    # Run the second Python file
    subprocess.run(['python', file2])