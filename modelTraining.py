from fastai.vision.all import *

train_path = 'ISIC-images-Training'
valid_path = 'ISIC-images-Validation'
train_df = pd.read_csv('challenge-2018-task-1-2-training_metadata_2024-09-20.csv')
valid_df = pd.read_csv('challenge-2018-task-3-validation_metadata_2024-09-20.csv')

train_df['image_path'] = train_path
valid_df['image_path'] = valid_path
df = pd.concat([train_df, valid_df], ignore_index=True)

df = df.dropna(subset=['benign_malignant']).reset_index(drop=True)

def image_exists(row):
    img_path = Path(f"{row['image_path']}/{row['isic_id']}.jpg")
    return img_path.exists()

df = df[df.apply(image_exists, axis=1)].reset_index(drop=True)

train_idx = list(range(len(train_df)))
valid_idx = list(range(len(valid_df), len(df)))


melanoma_data = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=lambda row: f"{row['image_path']}/{row['isic_id']}.jpg",  # Get the image file path
    get_y=ColReader('benign_malignant'),  # Get the labels
    splitter=RandomSplitter(valid_pct=0.2, seed=42),  # Split into training and validation sets
    item_tfms=Resize(224),  # Resize images to a consistent size
    batch_tfms=aug_transforms(flip_vert=True, max_rotate=20, max_zoom=1.1)
)

dls = melanoma_data.dataloaders(df)

dls.show_batch()

learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(6)
learn.show_results()