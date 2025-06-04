import torch
import pandas as pd
import os
from resnet_modev_training_v2 import ImageCaptionInference


test_csv_file = "/home/thkim/dev/eda/Toon_Persona_eda/toon_caption_metric_test_dataset.csv"
test_df = pd.read_csv(test_csv_file)
all_image_dir = "/HDD/toon_persona/Training/origin"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_type = 'resnet50'

inference = ImageCaptionInference("/home/thkim/dev/eda/Toon_Persona_eda/TaehongKim/model/best_resnet_caption_model_v2.pth", device, resnet_type)
caption_list = []
for idx, row in test_df.iterrows():
    image_path = os.path.join(all_image_dir, row['origin'])
    if os.path.exists(image_path):
        predicted_caption = inference.generate_caption(image_path)
        caption_list.append(predicted_caption)

pred_df = pd.DataFrame(test_df, columns=['origin','caption'])
pred_df['predict'] = caption_list
pred_df.to_csv("/home/thkim/dev/eda/Toon_Persona_eda/output.csv", index=False)
