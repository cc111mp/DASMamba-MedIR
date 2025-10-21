import os 
os.environ['CUDA_VISIBLE_DEVICES']='0'  

from evaluation.evaluation_metric import compute_measure
from data.common import transformData, dataIO
from data.MedicalDataUniform import Test_Data
import numpy as np
import torch
from torch.utils.data import DataLoader 
from tqdm import tqdm
import pandas as pd
from model.DASMamba import DASMamba

def load_model_for_inference(model, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Use 'model_state_dict' if available, else use the checkpoint directly
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        
        print(f"Model loaded successfully from {checkpoint_path}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
    
#############################################################################################################

transformData = transformData()
io = dataIO() 

# Paths setup
data_root = ""
modality_list = ["CT"] # ["PET", "CT", "MRI"]
save_dir = ""
checkpoint_path = ""

# Create directories
os.makedirs(save_dir, exist_ok=True)
save_result_dir = os.path.join(save_dir, "test_result")
os.makedirs(save_result_dir, exist_ok=True)

# Load model
Generator = DASMamba() 
Generator = load_model_for_inference(Generator, checkpoint_path)

if Generator is None:
    print("Failed to load model. Exiting...")
    exit()

for modality_name in modality_list:
    # Create modality-specific directory
    modality_save_dir = os.path.join(save_result_dir, modality_name)
    os.makedirs(modality_save_dir, exist_ok=True)
    
    # Initialize data loader
    test_loader = DataLoader(
        Test_Data(root_dir=data_root, modality_list=[modality_name], target_folder="test"),
        batch_size=1, 
        shuffle=False, 
        num_workers=4
    ) 
    
    # Initialize metric lists
    psnr_list, ssim_list, rmse_list, name_list = [], [], [], []
    
    # Testing loop
    for counter, data in enumerate(tqdm(test_loader, desc=f"Processing {modality_name}")):
        v_in_pic, v_label_pic, modality, file_name = data 
        modality = modality[0] 
        file_name = file_name[0] 
    
        v_in_pic = v_in_pic.type(torch.FloatTensor).cuda() 
        v_label_pic = v_label_pic.type(torch.FloatTensor) 
        
        # Generate output
        with torch.no_grad():
            gen_img = Generator(v_in_pic) 
        
        # Post-processing
        gen_img = transformData.denormalize(gen_img, modality).detach().cpu() 
        v_label_pic = transformData.denormalize(v_label_pic, modality) 
        
        # Truncation for test image (CT:[-160, 240])
        gen_img = transformData.truncate_test(gen_img, modality) 
        v_label_pic = transformData.truncate_test(v_label_pic, modality) 
        
        # Compute metrics
        data_range = v_label_pic.max() - v_label_pic.min()
        psnr, ssim, rmse = compute_measure(gen_img, v_label_pic, data_range=data_range)
        
        # Store results
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        rmse_list.append(rmse)
        name_list.append(file_name)

        # Save generated image
        save_path = os.path.join(modality_save_dir, f"{file_name}.nii")
        io.save(gen_img.clone().numpy().squeeze(), save_path)
    
    # Calculate final metrics
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list) 
    rmse_list = np.array(rmse_list)
    name_list = np.array(name_list)
    
    c_psnr = psnr_list.mean()
    c_ssim = ssim_list.mean()
    c_rmse = rmse_list.mean()
    
    print(f"\nFinal Test {modality_name}:")
    print(f"PSNR: {c_psnr:.4f}")
    print(f"SSIM: {c_ssim:.4f}")
    print(f"RMSE: {c_rmse:.4f}")
    
    # Save metrics to CSV
    result_dict = {
        "NAME": name_list,
        "PSNR": psnr_list,
        "SSIM": ssim_list, 
        "RMSE": rmse_list,
    }
    result = pd.DataFrame(result_dict)
    csv_path = os.path.join(save_result_dir, f"{modality_name}_result.csv")
    result.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")