📘 Transliteration with Seq2Seq + Attention

This project implements a Sequence-to-Sequence (Seq2Seq) model with Bahdanau Attention to perform transliteration on the Google Dakshina Dataset
Features

Encoder–Decoder architecture with RNN / GRU / LSTM (configurable).

Bahdanau Attention mechanism for better alignment between input & output characters.

Character-level transliteration (Latin → Devanagari).

Supports hyperparameter tuning (embedding size, hidden size, layers, cell type).

Training & evaluation with Exact Match accuracy.

Exports attention heatmaps to visualize how the model focuses on input characters.

Ready-to-use with Google Dakshina Hindi dataset.
├── translit_seq2seq.py       # Main training script
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies (optional)

## Installations
# Clone this repo
git clone https://github.com/<your-username>/seq2seq-transliteration.git
cd seq2seq-transliteration

# Install dependencies
pip install torch numpy matplotlib scikit-learn
