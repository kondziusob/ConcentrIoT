class Config(object):
    
    #predefined size of the image passed to the neural network 
    IMG_SIZE = (224,224)

    #, "Database/Masks/00001_mask.jpg","Database/Backgrounds/bg1.jpg"

    # directory with samples of the same object (in a sens of class, but not being our object we want to detect)
    SAMPLE_DIR = "Database/cars_train"
    
    #directory of foreground images = photos of the object we want to spont on images
    #FG_DIR = 'Database/Photos/Mycar'
    FG_DIR = "Database/VW/tmp3"
    
    #directory of preciously generated masks from photos from FG_DIR
    #MASK_DIR = 'Database/Masks'
    MASK_DIR = "Database/VW_Masks2"
    
    #directory of background images = random photos to diversified database
    #BG_DIR = 'Database/Backgrounds'
    BG_DIR = "Database/Backgrounds"

    #size of a butch used in training network on custom objects 
    BATCH_SIZE = 32
    
    NUM_BATCH_IN_EPOCH = 5
    
    #fraction of a training set which will be splited to validation and test dataset 
    VALIDATION_SPLIT = 0.2

    #random seed usef in shuffeling database 
    SEED = 1234

    INITIAL_EPOCHS = 3

    LR = 0.0001

    FINE_TUNE_EPOCHS = 5

    FINE_TUNE_AT = 100

    FINE_TUNE_LR = LR/10

    #SCALE = 1./127.5 
    
    #OFFSET = -1 