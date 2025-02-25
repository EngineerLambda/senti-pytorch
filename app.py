import streamlit as st
import torch
from torch import nn
import joblib
import os

st.title("Sentiment Prediction App- Model (LSTM) trained with PyTorch")
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded) # hidden is only needed here, the rest are just use when we need to keep the entire context in tasks like translation and text generation
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)
    
    
@st.cache_resource
def load_model():
    path = os.path.join(os.getcwd(), "resources")
    model = torch.load(os.path.join(path, "senti-model.pt"), map_location=torch.device('cpu'))
    vocab = joblib.load(os.path.join(path, "vocab.joblib"))
    le = joblib.load(os.path.join(path, "label_encoder.joblib"))
    
    return model, vocab, le

model, vocab, le = load_model()

def predict(text: str):
    tokens = text.split()
    token_ids = vocab(tokens)
    tensors = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        preds = model(tensors)
        
    pred_class = torch.argmax(preds, dim=1).item()
    text_class = le.inverse_transform([pred_class])[0]
    
    st.write(f"Predicted Class: {text_class}")
    
text = st.text_input("Enter text below")

if st.button("Determine Sentiment"):
    predict(text=text)