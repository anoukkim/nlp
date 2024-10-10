!pip install transformers[torch]
!pip install gradio
!git clone https://github.com/EX3exp/Kpop-lyric-datasets.git

from google.colab import drive
drive.mount('/gdrive', force_remount=True)

import os
import json
import pandas as pd
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

dataset_folder = '/content/Kpop-lyric-datasets/melon/monthly-chart'

def load_data(folder):
    dataframes = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    df = pd.json_normalize(data)
                    dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

df = load_data(dataset_folder)

if 'lyrics.lines' in df.columns:
    df['lyrics'] = df['lyrics.lines'].apply(lambda lines: ' '.join(lines) if isinstance(lines, list) else '')
else:
    raise ValueError("Column 'lyrics.lines' not found in the dataset.")

df = df[['song_name', 'artist', 'album', 'release_date', 'genre', 'lyrics']]
df = df.drop_duplicates(subset=['song_name', 'artist'])

top_5_genres = df['genre'].value_counts().index[:5].tolist()
print(f"Top 5 genres: {top_5_genres}")

df['genre'] = df['genre'].apply(lambda x: x if x in top_5_genres else 'Other')

print(df['genre'].value_counts())

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv('train_kpop_lyrics.csv', index=False)
val_df.to_csv('val_kpop_lyrics.csv', index=False)

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')

class LyricsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = dataframe.lyrics
        self.labels = dataframe.genre
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts.iloc[index]
        genre = self.labels.iloc[index]
        inputs = self.tokenizer.encode_plus(
            genre + " " + text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        ids = inputs["input_ids"].flatten()
        mask = inputs["attention_mask"].flatten()
        return {
            'input_ids': ids,
            'attention_mask': mask,
            'labels': ids
        }

train_dataset = LyricsDataset(train_df, tokenizer, max_len=512)
val_dataset = LyricsDataset(val_df, tokenizer, max_len=512)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

config = GPT2Config.from_pretrained("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2", config=config)

training_args = TrainingArguments(
    output_dir="/gdrive/MyDrive/KpopLyricsModel02",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("/gdrive/MyDrive/KpopLyricsModel02")
tokenizer.save_pretrained("/gdrive/MyDrive/KpopLyricsModel02")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

vectorizer = TfidfVectorizer().fit(df['lyrics'])

def generate_lyrics(genre, prompt_text, max_length=100):
    input_text = genre + " " + prompt_text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=max_length,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.0,
        temperature=1.0,
        num_return_sequences=1
    )

    generated_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    generated_text = generated_text[len(genre) + 1:]

    generated_vec = vectorizer.transform([generated_text])
    existing_vecs = vectorizer.transform(df['lyrics'])
    similarities = cosine_similarity(generated_vec, existing_vecs).flatten()

    top_5_indices = similarities.argsort()[-5:][::-1]
    top_5_songs = df.iloc[top_5_indices][['song_name', 'artist']]
    top_5_songs['similarity'] = similarities[top_5_indices]

    return generated_text, top_5_songs

genres = top_5_genres + ["Other"]

def gradio_interface(genre, prompt):
    generated_text, top_5_songs = generate_lyrics(genre, prompt)
    top_5_songs_str = ""
    for index, row in top_5_songs.iterrows():
        top_5_songs_str += f"{row['song_name']} by {row['artist']} (Similarity: {row['similarity']:.2f})\n"
    return generated_text, top_5_songs_str

gr_interface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Dropdown(choices=genres, label="Genre"), gr.Textbox(label="Input Text")],
    outputs=["text", "text"],
    title="노래 가사 생성 서비스"
)


gr_interface.launch()