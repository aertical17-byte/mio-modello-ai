# 🤖 Mio Modello AI

Un modello di linguaggio (GPT) addestrato **interamente nel cloud**, senza installare nulla sul tuo computer.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aertical17-byte/mio-modello-ai/blob/main/notebooks/addestra_modello.ipynb)

## ✨ Caratteristiche

- **100% cloud**: il training avviene su Google Colab (GPU gratuita T4). Nulla sul tuo PC.
- **Da zero**: il modello viene creato con pesi casuali e addestrato su un dataset reale.
- **Open source**: codice e notebook su GitHub, liberamente modificabili.
- **Salvataggio**: il modello addestrato può essere salvato su Hugging Face Hub o scaricato.

## 🚀 Come addestrare il modello (3 passi)

1. Clicca il badge **"Open in Colab"** qui sopra (o apri il notebook in `notebooks/addestra_modello.ipynb`).
2. In Colab: `Runtime → Change runtime type → T4 GPU` (gratis, nessuna carta di credito).
3. `Runtime → Run all` e aspetta ~10-20 minuti.

Il notebook:
- scarica il dataset `TinyStories` dal cloud,
- crea un GPT piccolo (~5M parametri) da zero,
- lo addestra sulla GPU gratuita di Google,
- genera testo di esempio e salva il modello.

## 📁 Struttura del repository

```
mio-modello-ai/
├── README.md
├── notebooks/
│   └── addestra_modello.ipynb   # notebook di training (apribile in Colab)
├── model_card.md                # scheda del modello
└── .gitignore
```

## 🔧 Personalizzazione

Apri il notebook e modifica la cella **"Configurazione"**:
- `N_TRAIN` / `N_VAL`: quanti esempi usare.
- `EPOCHS`: quante epoche di training.
- `N_LAYER`, `N_HEAD`, `N_EMBD`: dimensione del modello (più grande = più lento ma migliore).
- `DATASET`: qualsiasi dataset di testo su Hugging Face.

## 💾 Usare il modello addestrato

Dopo il training, in una cella Colab (o in un altro notebook):

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
model = GPT2LMHeadModel.from_pretrained("./mio-modello-ai-final")
tok = GPT2TokenizerFast.from_pretrained("./mio-modello-ai-final")
```

Oppure caricalo su Hugging Face Hub decommentando `notebook_login()` e `trainer.push_to_hub()`.

## 🌐 Piattaforme cloud gratuite per addestrare AI

- **Google Colab** (consigliato): GPU T4 gratuita, 15-30h/settimana.
- **Kaggle Notebooks**: GPU/TPU gratuiti.
- **Hugging Face Spaces (ZeroGPU)**: per demo e inferenza.
- **Lightning AI**: 80h/mese gratis.

## 📜 Licenza

MIT. Usa, modifica e condividi liberamente.
