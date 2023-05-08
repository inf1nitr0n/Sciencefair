Program uses neural network to autheticate signatures.

Create a folder named “samples” with a subfolder for each sample set (e.g. samples05, samples10, samples25, …). Each subfolder must contain a test and a train subfolder. Each of these subfolders must include both a subfolder with authentic signatures named 001 and a subfolder with non-authentic signatures named 001_forg. Use picture 1 for reference.

Choose a test dataset from the temporary folder of 25 authentic and 25 non-authentic randomly selected signatures from the temporary folder and store them into all test folders accordingly

Choose a train dataset from the temporary folder of 75 authentic and 75 non-authentic signatures different to the test dataset and store them into the train folder in samples75

Repeat step 6 with 50, 25, 10, 5 signatures

Access the neural network program with this QR code and import it into Visual Studio:

Set the value of the samples_directory in the Settings class to the pathname of the “samples” folder

Run the program of the neural network and determine the accuracy of the signature authentication of differently trained neural networks from the excel spreadsheet created when running the program .
