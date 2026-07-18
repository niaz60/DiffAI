
# XRD project for 2D    

| File      | Description |    
| ----------- | ----------- |     
| demo_train.py      | Train a model on 2D images from scratch.       |     
| demo_train_2D_pre_train.py   | Train a model on 2D images with ImageNet initialization.        |   
| demo_train_2D_pre_train_hkl.py   | Train a model on 2D images with ImageNet initialization, and the model outputs both the 230-way classification result and the HKL label.        |      
| demo_train_2D_pre_train_hkl_large7.py   | Train a model on 2D images with ImageNet initialization and the model outputs both the 230-way and 7-way classification result.        |      
| demo_train_zone1.py   | Train a model on 2D images generated on 1 zone axis.        |   
| demo_train_zone4.py   | Train a model on 2D images generated on 4 zone axis.        |   
| demo_train_zone10.py   | Train a model on 2D images generated on 10 zone axis.        |   
| demo_eval_2D.py   | Eval a model on 2D images.        |    
| demo_eval_2d_fair_4axis.py   | Eval a model on 2D images which are generated on 4 axis.        |    
| demo_eval_2d_fair_4axis.py   | Eval a model on 2D images which are generated on 10 axis.        |    
| demo_eval_2d_fair_4axis.py   | Eval a model on 2D images which are generated on 16 axis.        |    
| demo_eval_2D_multi_entropy.py   | Eval a model's uncertainty.        |    
| demo_eval_batch.py   | Batch evaluation (multi models and datasets).        |    
| dataloader.py   | Dataloader class.        |    
| generate_label.py   | generate labels saved as the csv file for a given batch of images.        |    