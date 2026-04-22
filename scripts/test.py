
import os
from matplotlib import pyplot as plt
import torch
from torchvision.models import efficientnet_b1
import sys

sys.path.append(os.getcwd())

from freehand.efficientnet_hook import EfficientNetFeatureRet
from freehand.loader import SSFrameDataset
from freehand.network import build_model
from data.calib import read_calib_matrices
from freehand.transform import LabelTransform, TransformAccumulation, PredictionTransform
from freehand.utils import *

print(torch.version.cuda)
print(torch.cuda.get_arch_list())
# GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.cuda.empty_cache()  # Clear GPU cache
else:
    device = torch.device('cpu')
    print("Warning: CUDA not available. Using CPU for training.")
    print("For GPU support, install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
print(f"Using device: {device}\n")

RESAMPLE_FACTOR = 4
FILENAME_CALIB = "data/calib_matrix.csv"
FILENAME_FRAMES = os.path.join(os.getcwd(), "data/Freehand_US_data", 'frames_res{}'.format(RESAMPLE_FACTOR)+".h5")

## algorithm parameters
PRED_TYPE = "parameter"  # {"transform", "parameter", "point"}
LABEL_TYPE = "point"  # {"point", "parameter"}
NUM_SAMPLES = 10
SAMPLE_RANGE = 10
NUM_PRED = 9
LEARNING_RATE = 1e-4

saved_results = 'seq_len' + str(NUM_SAMPLES) + '__' + 'lr' + str(LEARNING_RATE)\
        + '__pred_type_'+str(PRED_TYPE) + '__label_type_'+str(LABEL_TYPE) 
SAVE_PATH = os.path.join('results', saved_results)
if not os.path.exists(os.path.join(os.getcwd(),SAVE_PATH,'plotting')):
    os.makedirs(os.path.join(os.getcwd(),SAVE_PATH,'plotting'))
if not os.path.exists(os.path.join(os.getcwd(),SAVE_PATH,'pose_data')):
    os.makedirs(os.path.join(os.getcwd(),SAVE_PATH,'pose_data'))
if not os.path.exists(os.path.join(os.getcwd(),SAVE_PATH,'features')):
    os.makedirs(os.path.join(os.getcwd(),SAVE_PATH,'features'))
FILENAME_TEST = "fold_04.json"
FILENAME_WEIGHTS = "best_validation_dist_model"


## create the validation/test set loader
dset_test = SSFrameDataset.read_json(os.path.join(SAVE_PATH,FILENAME_TEST))
# TODO compare with train parameters before changing to all frames
if NUM_SAMPLES != dset_test.num_samples:
    raise("Inconsistent num_samples.")
if SAMPLE_RANGE != dset_test.sample_range:
    raise("Inconsistent sample_range.")
dset_test = SSFrameDataset.read_json(os.path.join(SAVE_PATH,FILENAME_TEST), num_samples=-1)

data_pairs = pair_samples(NUM_SAMPLES, NUM_PRED)
frame_points = reference_image_points(dset_test.frame_size,2).to(device)
tform_calib = torch.tensor(read_calib_matrices(
    filename_calib=FILENAME_CALIB, 
    resample_factor=RESAMPLE_FACTOR
    ), device=device)


transform_prediction = PredictionTransform(
    PRED_TYPE, 
    'transform', 
    num_pairs=data_pairs.shape[0], 
    image_points=frame_points, 
    tform_image_to_tool=tform_calib
    )
accumulate_prediction = TransformAccumulation(
    image_points=frame_points, 
    tform_image_to_tool=tform_calib
    )

pred_dim = type_dim(PRED_TYPE, frame_points.shape[1], data_pairs.shape[0])
label_dim = type_dim(LABEL_TYPE, frame_points.shape[1], data_pairs.shape[0])


## load the model
model = build_model(
    efficientnet_b1, 
    in_frames = NUM_SAMPLES, 
    out_dim = pred_dim
    ).to(device)
model.load_state_dict(torch.load(os.path.join(SAVE_PATH,'saved_model',FILENAME_WEIGHTS), map_location=torch.device(device)))
model.train(False)

model = EfficientNetFeatureRet(model)


## inference
for i_scan in range(len(dset_test)):
        
    SCAN_INDEX = i_scan  # plot one scan
    PAIR_INDEX = 0  # which prediction to use
    START_FRAME_INDEX = 0  # starting frame - the reference 

    # load frames
    frames, tforms, tforms_inv = dset_test[SCAN_INDEX]
    frames, tforms, tforms_inv = (torch.tensor(t).to(device) for t in [frames,tforms,tforms_inv])
    
    # prepare predictions and data pairs for transformation
    predictions_allpts = torch.zeros((frames.shape[0],3,frame_points.shape[-1]), device=device)
    predictions_alltransforms = torch.zeros((frames.shape[0],4,4), device=device)
    features_allpts = [] # store as list as feature dimension is not determined

    data_pairs_all = data_pairs_cal_label(frames.shape[0])
    transform_label = LabelTransform(
            "point",
            pairs=data_pairs_all,
            image_points=frame_points,
            tform_image_to_tool=tform_calib
            )
    
    # prepare GT labels
    labels_allpts = torch.squeeze(transform_label(tforms[None,...], tforms_inv[None,...]))
    
    # 
    idx_f0 = START_FRAME_INDEX # this is the reference starting frame for network prediction 
    #idx_p0 = idx_f0 + data_pairs[PAIR_INDEX][0] # this is the reference frame for transforming others to
    idx_p1 = idx_f0 + data_pairs[PAIR_INDEX][1]
    interval_pred = data_pairs[PAIR_INDEX][1] - data_pairs[PAIR_INDEX][0]


    tform_1to0 = torch.eye(4, device=device)
    predictions_allpts[idx_f0+1] = labels_allpts[0]

    while 1:
        frames_test = frames[idx_f0:idx_f0+NUM_SAMPLES,...]
        frames_test = frames_test/255
        outputs_test, features_test = model(frames_test.unsqueeze(0))
        features_allpts.append(features_test)

        tform_2to1 = transform_prediction(outputs_test)[0,PAIR_INDEX]
        preds_val, tform_1to0 = accumulate_prediction(tform_1to0, tform_2to1)
        predictions_allpts[idx_f0+1] = preds_val
        predictions_alltransforms[idx_f0+1] = tform_2to1
        
        idx_f0 += interval_pred
        idx_p1 += interval_pred 
        if (idx_f0+NUM_SAMPLES) > frames.shape[0]:
            break
    if NUM_SAMPLES > 2:
        predictions_allpts[idx_f0:,...] = predictions_allpts[idx_f0-1].expand(predictions_allpts[idx_f0:,...].shape[0],-1,-1)
        features_allpts.extend([features_allpts[idx_f0 - 1]] * (predictions_allpts.shape[0] - len(features_allpts)))
        predictions_alltransforms[idx_f0:,...] = predictions_alltransforms[idx_f0-1].expand(predictions_alltransforms[idx_f0:,...].shape[0],-1,-1)

    # plot trajectory
    scan_plot_gt_pred(
        labels_allpts.detach().cpu().numpy(),
        predictions_allpts.detach().cpu().numpy(),
        SAVE_PATH +'/'+'plotting'+'/' + str(i_scan),
        color='g',
        width=4,
        scatter=8,
        legend_size=50,
        legend='GT'
    )    

    break
# save pose and transform data
torch.save(predictions_allpts.detach().cpu(), os.path.join(SAVE_PATH +'/'+'pose_data', 'predictions.pt'))
torch.save(labels_allpts.detach().cpu(), os.path.join(SAVE_PATH +'/'+'pose_data', 'labels.pt'))
torch.save(predictions_alltransforms.detach().cpu(), os.path.join(SAVE_PATH +'/'+'pose_data', 'predictions_transforms.pt'))

# save features
features_allpts = torch.stack(features_allpts, dim=0)
torch.save(features_allpts.detach().cpu(), os.path.join(SAVE_PATH +'/'+'features','features_allpts.pt'))

print("Transform predictions shape:", predictions_alltransforms.shape)
print("features shape:", len(features_allpts))
print("labels_allpts shape:", labels_allpts.shape)
print("predictions_allpts shape:", predictions_allpts.shape)
