import json
import os

notebook_path = r"c:/Users/thali/Downloads/chest-xray-frontend (1)/chest-xray-frontend/orginal1st (9).ipynb"

# New code to replace the incorrect implementations
new_code = [
    "\n",
    "# ============================================================================\n",
    "# REAL GRADCAM ENDPOINT IMPLEMENTATION (CORRECTED FOR FRONTEND)\n",
    "# ============================================================================\n",
    "\n",
    "def generate_gradcam_data(model, model_name, image_bytes):\n",
    "    \"\"\"\n",
    "    Generates GradCAM data (heatmap, overlay) and analysis.\n",
    "    \"\"\"\n",
    "    try:\n",
    "        # Load image\n",
    "        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')\n",
    "        \n",
    "        # Create a visualization transform that matches the model's spatial transform\n",
    "        # but keeps it as an image (no normalization, no tensor conversion yet)\n",
    "        viz_transform = T.Compose([\n",
    "            T.Resize((256, 256)),\n",
    "            T.CenterCrop(224)\n",
    "        ])\n",
    "        \n",
    "        # 1. Prepare image for Visualization (Aligned with Model Input)\n",
    "        viz_img = viz_transform(img) # 224x224 PIL Image (Cropped)\n",
    "        img_array = np.array(viz_img) # 224x224 Numpy array\n",
    "        \n",
    "        # 2. Prepare image for Model (Tensor + Normalization)\n",
    "        tensor_transform = T.Compose([\n",
    "            T.ToTensor(),\n",
    "            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])\n",
    "        ])\n",
    "        img_tensor = tensor_transform(viz_img).unsqueeze(0).to(device)\n",
    "\n",
    "        # Enable gradients\n",
    "        model.eval()\n",
    "        for param in model.parameters():\n",
    "            param.requires_grad = True\n",
    "\n",
    "        # Target layers\n",
    "        target_layers = None\n",
    "        if 'DenseNet' in model_name:\n",
    "            if hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):\n",
    "                 target_layers = [model.features.denseblock4]\n",
    "            else:\n",
    "                 target_layers = [model.features[-1]]\n",
    "        elif 'ResNet' in model_name:\n",
    "            target_layers = [model.layer4[-1]]\n",
    "        elif 'EfficientNet' in model_name:\n",
    "             if hasattr(model, 'conv_head'): target_layers = [model.conv_head]\n",
    "             elif hasattr(model, 'blocks'): target_layers = [model.blocks[-1]]\n",
    "        \n",
    "        if target_layers is None:\n",
    "            for module in reversed(list(model.modules())):\n",
    "                if isinstance(module, nn.Conv2d):\n",
    "                    target_layers = [module]\n",
    "                    break\n",
    "        \n",
    "        if target_layers is None:\n",
    "            print(f\"Could not find target layer for {model_name}\")\n",
    "            return None\n",
    "\n",
    "        # Prediction\n",
    "        with torch.no_grad():\n",
    "            outputs = model(img_tensor)\n",
    "            probs = torch.softmax(outputs, dim=1)\n",
    "            confidence, pred_idx = probs.max(1)\n",
    "            pred_idx = pred_idx.item()\n",
    "            confidence = confidence.item()\n",
    "            predicted_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else \"Unknown\"\n",
    "\n",
    "        # GradCAM\n",
    "        from pytorch_grad_cam import GradCAM\n",
    "        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget\n",
    "        \n",
    "        cam = GradCAM(model=model, target_layers=target_layers)\n",
    "        targets = [ClassifierOutputTarget(pred_idx)]\n",
    "        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]\n",
    "        \n",
    "        # Heatmap\n",
    "        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)\n",
    "        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)\n",
    "        \n",
    "        # Overlay\n",
    "        overlay = (0.4 * heatmap + 0.6 * img_array).astype(np.uint8)\n",
    "        \n",
    "        # Focus Areas Extraction (Mathematical Results)\n",
    "        threshold = 0.7 * np.max(grayscale_cam)\n",
    "        _, binary = cv2.threshold(np.uint8(255 * grayscale_cam), 255 * 0.7, 255, cv2.THRESH_BINARY)\n",
    "        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
    "        \n",
    "        important_regions = []\n",
    "        for i, contour in enumerate(contours):\n",
    "            x, y, w, h = cv2.boundingRect(contour)\n",
    "            mask = np.zeros_like(grayscale_cam)\n",
    "            cv2.drawContours(mask, [contour], 0, 1, -1)\n",
    "            mean_intensity = np.mean(grayscale_cam[mask == 1])\n",
    "            \n",
    "            important_regions.append({\n",
    "                \"Region\": f\"#{i+1}\",\n",
    "                \"Intensity\": f\"{mean_intensity:.2f}\",\n",
    "                \"Location\": f\"x={x}, y={y}\",\n",
    "                \"Size\": f\"{w}x{h}\"\n",
    "            })\n",
    "        \n",
    "        # Convert to Base64\n",
    "        def to_b64(img_arr):\n",
    "            img = Image.fromarray(img_arr)\n",
    "            buff = io.BytesIO()\n",
    "            img.save(buff, format=\"PNG\")\n",
    "            return base64.b64encode(buff.getvalue()).decode('utf-8')\n",
    "            \n",
    "        return {\n",
    "            \"heatmap\": to_b64(heatmap),\n",
    "            \"overlay\": to_b64(overlay),\n",
    "            \"prediction\": predicted_class,\n",
    "            \"confidence\": confidence,\n",
    "            \"important_regions\": important_regions,\n",
    "            \"probabilities\": {CLASS_NAMES[i]: float(probs[0][i]) for i in range(min(len(CLASS_NAMES), len(probs[0])))}\n",
    "        }\n",
    "    except Exception as e:\n",
    "        print(f\"Error in generate_gradcam_data: {e}\")\n",
    "        import traceback\n",
    "        traceback.print_exc()\n",
    "        return None\n",
    "\n",
    "@app.post(\"/gradcam\")\n",
    "async def generate_gradcam_endpoint(\n",
    "    file: UploadFile = File(...),\n",
    "    model_name: str = Form(\"DenseNet169\")\n",
    "):\n",
    "    try:\n",
    "        image_bytes = await file.read()\n",
    "        \n",
    "        # Handle model selection\n",
    "        if model_name not in trained_models:\n",
    "             found = False\n",
    "             for name in trained_models.keys():\n",
    "                 if model_name.lower() in name.lower():\n",
    "                     model_name = name\n",
    "                     found = True\n",
    "                     break\n",
    "             if not found:\n",
    "                 model_name = list(trained_models.keys())[0]\n",
    "             \n",
    "        model = trained_models[model_name]\n",
    "        \n",
    "        data = generate_gradcam_data(model, model_name, image_bytes)\n",
    "        if not data:\n",
    "            raise HTTPException(500, \"Failed to generate GradCAM\")\n",
    "            \n",
    "        # Return FLAT structure matching GradCAMViewer.jsx\n",
    "        return {\n",
    "            \"status\": \"success\",\n",
    "            \"model_used\": model_name,\n",
    "            \"heatmap\": data[\"heatmap\"],\n",
    "            \"overlay\": data[\"overlay\"],\n",
    "            \"prediction\": data[\"prediction\"], # String\n",
    "            \"confidence\": data[\"confidence\"], # Float\n",
    "            \"probabilities\": data[\"probabilities\"],\n",
    "            \"important_regions\": data[\"important_regions\"],\n",
    "            \"explanation\": f\"The model detected {data['prediction']} with {data['confidence']:.1%} confidence. The heatmap highlights the regions contributing most to this decision.\",\n",
    "            \"timestamp\": datetime.now().isoformat()\n",
    "        }\n",
    "    except Exception as e:\n",
    "        print(f\"Endpoint error: {e}\")\n",
    "        raise HTTPException(500, str(e))\n",
    "\n",
    "@app.post(\"/gradcam/test\")\n",
    "async def test_gradcam_response(\n",
    "    file: UploadFile = File(...),\n",
    "    model_name: str = Form(\"DenseNet169\")\n",
    "):\n",
    "    \"\"\"\n",
    "    Test endpoint with CORRECT structure for Frontend\n",
    "    \"\"\"\n",
    "    try:\n",
    "        # Create mock images\n",
    "        def create_mock_image(color):\n",
    "            img = Image.new('RGB', (224, 224), color)\n",
    "            buff = io.BytesIO()\n",
    "            img.save(buff, format=\"PNG\")\n",
    "            return base64.b64encode(buff.getvalue()).decode('utf-8')\n",
    "\n",
    "        heatmap_b64 = create_mock_image((255, 0, 0))\n",
    "        overlay_b64 = create_mock_image((255, 128, 0))\n",
    "\n",
    "        return {\n",
    "            \"status\": \"success\",\n",
    "            \"model_used\": model_name,\n",
    "            \"heatmap\": heatmap_b64,\n",
    "            \"overlay\": overlay_b64,\n",
    "            \"prediction\": \"COVID-19\",\n",
    "            \"confidence\": 0.95,\n",
    "            \"probabilities\": {\"COVID-19\": 0.95, \"Normal\": 0.05},\n",
    "            \"important_regions\": [\n",
    "                {\"Region\": \"#1\", \"Intensity\": \"0.85\", \"Location\": \"x=50, y=50\", \"Size\": \"40x40\"}\n",
    "            ],\n",
    "            \"explanation\": \"Mock response with correct flat structure.\",\n",
    "            \"timestamp\": datetime.now().isoformat()\n",
    "        }\n",
    "    except Exception as e:\n",
    "        return JSONResponse(status_code=500, content={\"error\": str(e)})\n"
]

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

source = nb['cells'][0]['source']

# Remove old implementations
# We will look for the start of my previous injection and the start of the old test endpoint
start_marker = "# REAL GRADCAM ENDPOINT IMPLEMENTATION"
test_marker = "@app.post(\"/gradcam/test\")"

start_idx = -1
test_idx = -1

for i, line in enumerate(source):
    if start_marker in line:
        start_idx = i
    if test_marker in line:
        test_idx = i

# If we found the start of my previous code, we remove everything from there up to the end of the cell or next section
# But wait, I want to replace the old test endpoint too.
# So I should remove from start_idx (if found) OR from test_idx (if start_idx not found, which shouldn't happen if I injected it)
# Actually, I should just find where I want to insert and remove the conflicting parts.

# Let's just remove the old test endpoint and my previous injection if present.
# My previous injection started with "# REAL GRADCAM ENDPOINT IMPLEMENTATION"
# The old test endpoint started with "@app.post(\"/gradcam/test\")"

# Strategy: Filter out lines that belong to these sections and append new code.
# This is risky if I cut too much.

# Better strategy:
# 1. Find index of "# REAL GRADCAM ENDPOINT IMPLEMENTATION" -> Remove from here until... where?
#    My previous injection ended before the test endpoint.
# 2. Find index of "@app.post(\"/gradcam/test\")" -> Remove this function block.

# Let's try to find the range of lines to delete.
# My previous code was inserted before /gradcam/test.
# So if I find start_marker, I can delete from there.
# And I also want to delete the OLD /gradcam/test which follows it (or was there before).

# Let's just look for the markers and rebuild the source list.
new_source = []
skip = False
skip_until = None

# We want to remove:
# 1. The block starting with "# REAL GRADCAM ENDPOINT IMPLEMENTATION"
# 2. The block starting with "@app.post(\"/gradcam/test\")" (and its function)
# 3. The block starting with "# ============================================================================\n# COMPARISON: Wrong vs Right" (which was part of the old test code)

# Actually, I can just keep everything UP TO the point where I started messing with it, or where the test endpoint starts.
# The test endpoint is usually near the end.

# Let's find the first occurrence of either marker.
cut_index = len(source)
for i, line in enumerate(source):
    if start_marker in line:
        cut_index = min(cut_index, i)
    if "@app.post(\"/gradcam/test\")" in line:
        cut_index = min(cut_index, i)

print(f"Truncating source at line {cut_index}")
nb['cells'][0]['source'] = source[:cut_index] + new_code

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with corrected GradCAM implementation.")
