# Model Card: mio-modello-ai

## Dettagli del modello
- **Architettura**: GPT-2 (causal language model)
- **Parametri**: ~5 milioni (4 layer, 4 head, embedding 256)
- **Tokenizer**: GPT-2 (pre-addestrato)
- **Context length**: 128 token
- **Addestramento**: da zero (pesi casuali) su TinyStories

## Training
- **Dataset**: roneneldan/TinyStories (subset)
- **Hardware**: Google Colab T4 GPU (gratis)
- **Framework**: Hugging Face Transformers + PyTorch
- **Ottimizzatore**: AdamW, lr=5e-4

## Uso previsto
Generazione di testo breve (storie semplici). Modello didattico, non per produzione.

## Limitazioni
Modello molto piccolo, addestrato su pochi dati e poche epoche. Le generazioni possono essere incoerenti o ripetitive. Solo inglese (dataset TinyStories).

## Come riprodurre
Apri `notebooks/addestra_modello.ipynb` in Google Colab con GPU T4 e clicca "Run all".
