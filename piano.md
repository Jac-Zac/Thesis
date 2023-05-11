# Piano per il progetto

### Idea generale divisa in step:
> I seguenti punti verrano giustificati brevemente sotto

1. Concentrarci inizialmente sul tagliare il testo dall'immagine, prima ritagliare solo le labels (anche grossolanamente ma tenendo tutto il testo), poi concentrarci sul fare `line segmentation`, questo non con tecniche di deep learning a meno che non siano tools già presenti. I think I will use something like this for line segmentation: [LayoutLMv3](https://arxiv.org/pdf/2204.08387.pdf). Thought I'd like something else because It cant be used for real products. But this is a replaceable part of the stack
Text segmentation also can use [MMOCR](https://github.com/open-mmlab/mmocr)


> Se questo primo step è fattibile anche non ottenendo risultati perfetti su tutto il dataset direi che si può continuare e focalizzarsi sugli step successivi

> Se invece questa parte si rivela troppo problematica si può pensare di allenare un modello da zero che invece di prendere linee di testo prenda un'intera immagine, (oppure ci sono potenziali altri modelli su cui dovrei informarmi di più)

2. Usare il base model [TrOCR](https://arxiv.org/pdf/2109.10282.pdf) e vedere che risultati otteniamo.

3. Fare Fine-tuning con i nostri dati e volendo anche altri dataset trovati online per esempio di Erbari.

3. A questo punto vedere di capire meglio come integrare NER (Name Entity Recognition) nel nostro stack. Quello che spero di fare sarebbe riuscire ad rimpiazzare i layer finali di TrOCR e fare transfer learning per il nostro task nello specifico.

### Giustificazioni

1. Sostanzialmente l'idea di estrarre solo la parte rilevante dell' immagine per il riconoscimento dovrebbe dare risultati migliori, forse ho anche visto un articolo in cui facevano la stessa cosa ma comunque è decisamente più sensato che fare una cosa End to End partendo dall'immagine a mio parere, anche se in futuro si potrebbe sempre provare un approccio di questo tipo

2. Usare TrOCR o un architettura similare (ViT [Visual Transform](https://arxiv.org/pdf/2010.11929.pdf)). Per più ragioni:
    - Migliori performance come si vede in [questo articolo](https://arxiv.org/pdf/2203.11008.pdf) per Historical Documents
    - (Parere personale) Quello che un encoder-decoder transformer impara è più simile al task successivo di NER e potrebbe inoltre essere meglio incorporato con qualcosa tipo [BERT](https://arxiv.org/pdf/1810.04805.pdf)

---
##### Per quanto riguarda i possibili problemi con la line segmentation:

Potenzialmente si potrebbe pensare ad un alternativo modello che non richiede line segmentation, però io preferirei rimanere comunque un ViT e non cose come CNN o LSTM (anche perché TrOCR è SOTA).

#### Potenziale materiale da esplorare [this for recognition of text](https://arxiv.org/abs/2104.07787)

### Materiali che ho guardato o non ho finito ma sono interessanti

- [X] Ho letto questa [Systematic Literature Review](https://arxiv.org/ftp/arxiv/papers/2104/2104.08732.pdf), la parte importante è a pagina 23, 24. (però mostra cose vecchie per esempio uso di LSTM quando io vorrei usare Transformers)

- [X] Da leggere con più attenzione, [TrOCR on Historical Documents](https://arxiv.org/pdf/2203.11008.pdf)

- [ ] Articolo che sembra anche interessante per un workflow generale: [Towards a scientific workflow + NLP](https://riojournal.com/article/55789/element/4/5731002//)

- [X] Trovare il testo in un [herbarium con YOLO](https://www.researchgate.net/publication/340039970_Objects_Detection_from_Digitized_Herbarium_Specimen_based_on_Improved_YOLO_V3) anche se non voglio focalizzarmi su questo dato che i dati per fare questa parte non sono quello che abbiamo

- [X] Questo spiega un [framework generale](https://docs.google.com/viewerng/viewer?url=https://digital.csic.es/bitstream/10261/239620/1/814319.pdf) poco interessante

- [X] Letto ma da rileggere per ragionarci su, sopratutto sulla parte finale: [`Comprensive Blog post on HTR`](https://nanonets.com/blog/handwritten-character-recognition/)


### Other idea:

[Donut really good](https://huggingface.co/docs/transformers/main/en/model_doc/donut)

#### Or I could just write my model taking:

[Swin](https://www.youtube.com/watch?v=SndHALawoag)

And then something like BERT or LLAMA7B and use that to train the entire stack

[SegFormer sota](https://www.youtube.com/watch?v=cgq2d_HkfnM)

Text is text: `https://www.youtube.com/watch?v=8VLkaf_hGdQ`

### Initial plan:

- Start out by trying [Donut](https://huggingface.co/docs/transformers/main/en/model_doc/donut)

- Than you can train it on your data with the token <extract-importants>

- Then you can instead retrain it on HTR with <extract>

#### Or switch to Donut following [this](https://www.philschmid.de/fine-tuning-donut)

- Potentially do some LORA fine-tuning or something like that on Donut or just build another model to detect a box around the text we want to use and then use it to cut the images automatically as a second step to make performances better

- I also believe that the accuracy will be pretty low if we look for exact string matching. And as the model gets better we would get to see an emergent jump in performance similarly to what is suggested in the paper: ```https://arxiv.org/pdf/2304.15004.pdf``` Thus I believe we shouldn't evaluate the model only on the exact match but have a more continues  way to evaluate it to get a more realistic accuracy and scaling for the future models. Look at something like (Token Edit Distance)

- UBIAI https://ubiai.tools/Docs#modelcreation for data annotation

- Read this for segmentation: https://www.researchgate.net/publication/340039970_Objects_Detection_from_Digitized_Herbarium_Specimen_based_on_Improved_YOLO_V3

- Use this dataset: ```herbarium Haussknecht of FSU Jena.``` for further training and use more then YOLO v4

- For all the predicted words I want to see the attention map to where they point and maybe do some noice visualization
