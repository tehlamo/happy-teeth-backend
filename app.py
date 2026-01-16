import os
import io
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

# ------------------------------------------------------------------------
# 1. CONFIGURATION & CUSTOM TEXT
# ------------------------------------------------------------------------

CLASS_NAMES = [
    "background", 
    "caries", 
    "gingivitis", 
    "tooth discoloration", 
    "ulcer", 
    "calculus", 
    "plaque", 
    "hypodontia"
]

HEALTH_TIPS = {
    "caries": "There seems to be a potential decay in your teeth, also known as caries. To reverse this or prevent this from worsening, try not too drink or snack too often! If it develops into a cavity, you must go to the dentist and get it filled.", 
    "gingivitis": "Seems like you have some redness around your gums, which points to gingivitis. It's best to remember to brush your teeth twice a day THOROUGHLY, and use a soft tooth brush, it's better! Don't forget to floss as well. If possible, try getting your teeth professionally cleaned by a dentist.", 
    "tooth discoloration": "Seems like your teeth aren't the color they should be, which are signs of tooth discoloration. If it's yellow, it could be food and drinks, aging, or signs you need to improve your oral hygeine. If it's brown, it might be due to drug use like tabacco or untreated tooth decay. If it's purple, it could be from drinking wine. If it's gray, it may mean the nerve inside of your tooth has died. If there's white flecks, it could mean you had high levels of flouride when you were younger, or you might begin forming cavities. Finally, black spots indicate severe tooth decay. If you're unsure abuot the color, do your best to contact a dentist!", 
    "ulcer": "If you see a white sore on the inside of your mouth, it's an ulcer or a lesion, and one of the best ways to alleviate them is with antiseptic gels. On the other hand, xerostomia is a lack of saliva in the mouth, and there are many reasons for it. The best you could do is breathe through your nose and drink lots of water. If you have any more questions on xerostomia, feel free to try and contact a dentist.", 
    "calculus": "Calculus is also known as tartar, and it could be caused by both forgetting to brush your teeth or forgetting to floss as often as you should. It could also come from sugary foods, tobacco, braces, or a dry mouth. The best thing to do is to go to a dentist and get a dental cleaning or get gum disease treatements.", 
    "plaque": "This is just a stage before tartar, being a softer version called plaque. To prevent plaque from worsening, brush your teeth twice daily, floss once daily, eat healthy foods, and see a dentist if you still have concerns.", 
    "hypodontia": "There seems to be missing teeth, which points to the condition called hypodontia. Although there isn't a way to prevent this often times, seeing a dentist for braces could help you, and make those teeth happy.", 
    "default": "It seems as though we weren't able to detect anything. If you suspect your oral health has issues, it might be best to ask a dentist, otherwise, your teeth seem healthy to us!"
}

MEDICAL_DISCLAIMER = "NOTE: These analyses on teeth were done by a oral disease model that detects certain defects in the images provided. The tips provided are not going to always help, and were personally made by Donovan Thach by reading through many articles. If you have the money, please do go to a dentist and get your teeth checked that way. This is just to clarify and try to prevent dental diseases before they worsen. Remember, always do your research when looking into health."

# ------------------------------------------------------------------------
# 2. MODEL LOADING
# ------------------------------------------------------------------------

app = FastAPI(title="Happy Teeth AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = None

def load_model():
    global model
    try:
        num_classes = len(CLASS_NAMES)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Load weights
        model_path = "best_model.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print("--- Happy Teeth Model Loaded ---")
        else:
            print("WARNING: best_model.pth not found!")
            
    except Exception as e:
        print(f"Error loading model: {e}")

load_model()

# ------------------------------------------------------------------------
# 3. PREDICTION ENDPOINT
# ------------------------------------------------------------------------

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 images allowed.")

    batch_results = []

    for file in files:
        try:
            # Read Image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # Transform
            transform = torchvision.transforms.ToTensor()
            image_tensor = transform(image).to(device)
            image_tensor = image_tensor.unsqueeze(0) 

            # Inference
            with torch.no_grad():
                prediction = model(image_tensor)[0]

            # Parse Predictions
            detections = []
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()

            for i in range(len(boxes)):
                score = float(scores[i])
                if score >= 0.5: # Confidence Threshold
                    class_id = int(labels[i])
                    
                    # Safe lookup for class name
                    if class_id < len(CLASS_NAMES):
                        class_name = CLASS_NAMES[class_id]
                    else:
                        class_name = "Unknown"
                    
                    # Safe lookup for tip
                    tip = HEALTH_TIPS.get(class_name, HEALTH_TIPS["default"])
                    
                    detections.append({
                        "label": class_name,
                        "confidence": round(score, 2),
                        "box": boxes[i].tolist(),
                        "tip": tip
                    })

            batch_results.append({
                "filename": file.filename,
                "status": "success",
                "detections": detections
            })

        except Exception as e:
            batch_results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })

    return {
        "disclaimer": MEDICAL_DISCLAIMER,
        "results": batch_results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)