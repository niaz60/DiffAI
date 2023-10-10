# XRD project

## How to init the project  
1. Create a empty foler and enter the created foler.
2. conduct the following commands,
````
git init .
git remote add origin git@github.com:ZhangAIPI/XRD_project.git
git pull
git checkout -b main origin/main
````
Then, we finish the repo initialization.   

## How to upload your code
1. Enter the main folder of the project.  
2. conduct the following commands,
````
git pull
git add .
git commit -m "say something about your updates"
git push
````
Please use the pull command to make your code being updated, then push your code. Otherwise, there might be some conflicts or errors :(  


## How to comment or give some suggestions about the code/progress/result?
1. Find the ___Issues___ button on this page and open it.  
2. Click the __New issue__ button and write something.  It is like a post. We can discuss about the submitted issues here. 
I also provide a example of issue in the Issues channel.   

## Progress sync
:white_check_mark: Reorganize the code for 1D XRD generation    
:white_large_square: 2D XRD generation    
:white_large_square: auto augmentation    


## How to run the pipeline?  
1. prepare your .cif files in the foler "CIFs_examples".  
2. conduct the following command,  
````
python run cif_pipeline.py
````  
3. The transformed data is saved in the foler "XRD_output_examples".  


## Tasks to be implemented in data generation codes   
:white_check_mark: Multi-threads to read data, one thread for reading and transforming one .cif file.   
:white_large_square: Partition the data processing task into multi sub-tasks on different processors. Different process will spawn multiple threads to read and process single .cif file independently and in paralle.      
:white_check_mark: wrap the aforementioned data generation code into the Dataloader.get_item().   
:white_check_mark: We can control the augmentation of data by passing different values of U, V, W to the get_item() function.  
:white_check_mark: Apply our Dataloader to the model training pipeline.  


We uploaded the pre-trained models in https://drive.google.com/drive/folders/1gDmfmgV0u3lZRSa-W7Z2nJpXb0wQOpnD?usp=sharing