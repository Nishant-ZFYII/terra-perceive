"""
segmentation_node.py — SegFormer semantic segmentation ROS2 node.

This is a LIBRARY CALL — you are NOT reimplementing SegFormer.
Your value: fine-tuning on RELLIS-3D + choosing the right architecture.

TO DO
- Export to ONNX for Jetson deployment

PDF reference: Part 7.1
Train ref: huggingface.co/docs/transformers/model_doc/segformer
Dataset: RELLIS-3D (Jiang et al., RA-L 2021)

 The RELLIS-3D label IDs (0, 3, 4...) are the ground truth annotation class IDs. But nvidia/segformer-b0-finetuned-ade-512-512  
  is trained on ADE20K — 150 completely different classes with different IDs.

will be using the pretrained SegFormer model nvidia/segformer-b0-finetuned-ade-512-512, which is trained on ADE20K. 
The RELLIS-3D dataset has 20 terrain classes with different IDs. 
To fine-tune the SegFormer model on RELLIS-3D, we need to map the RELLIS-3D class IDs to the corresponding ADE20K class IDs. 
This involves creating a mapping between the two sets of class IDs and then using this mapping during the 
fine-tuning process to ensure that the model learns to predict the correct classes for the RELLIS-3D dataset.
This will be done in phase 3.

"""
from transformers import  SegformerForSemanticSegmentation,AutoImageProcessor
import numpy as np
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import os


image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

#disable dropout layers
model.eval()

def run_segformer(image_path: str) -> np.ndarray:
    """ Returns HxW uint8 labela rray."""

    #load image
    image = Image.open(image_path).convert("RGB")

    #image preprocessing
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        #model pass
        outputs = model(**inputs)

        #extract logits
        #(1,150,H/4,W/4) for SegFormer-B0
        logits = outputs.logits

        #postprocess logits into labels

        width, height = image.size

        # upsample logits to original size

        upsampled_logits =  torch.nn.functional.interpolate(
                            input=logits, size=(height, width), mode='bilinear', align_corners=False)
        
        #argmax over class dimension
        labels = torch.argmax(upsampled_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        return labels
        
if __name__ == "__main__":
    image_path = "/home/nishant/MS_Project/terra-perceive/third_party/RELLIS-3D/utils/example/frame000104-1581624663_149.jpg"
    
    #inference
    labels = run_segformer(image_path)
    print("Unique predicted class IDs: %s" % np.unique(labels))
    print("Lable map shape (HxW): width = %s height = %s" % (labels.shape[1], labels.shape[0]))
    
    # load original image 
    original_image = Image.open(image_path).convert("RGB")

    #colorize label map for visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].imshow(original_image)                                                                                                   
    axes[0].set_title("Original image")
    axes[0].axis("off")                                                                                                        
    axes[1].imshow(labels, cmap="tab20")                                                                                       
    axes[1].set_title("SegFormer ADE20K predictions")
    axes[1].axis("off") 

    #save 
    out_path = "docs/assets/segformer_output.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)                                                                      
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)                                                                                             
    print(f"Saved → {out_path}")                                                                                               
    plt.show()

    