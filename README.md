# voice_assistant

## What is this?
Homemade vocal assistant, to practice using wav2vec2 from huggingface and class oriented programming.

After initialisation, the assistant can enter standby, waiting for some sound. If sound is detected, recording start until sound becomes sufficiently quiet for some time. The sound array is passed to a french-optimized wav2vec2 model, producing text. The text is then converted to a phonetic representation (for better robustness against false detection, such as "dis", "dit" or "dix") and comparted against a known database of orders.

## How does it work?

### prerequisites
todo

### installation

todo

### basic operation

va = assistant()

va.standby()

## To-do list

- Code missing functions (such as warmup of the model)
- allow for activation + order in the same speech
- properly reference source material (models, sound recording code, ...)
- 
