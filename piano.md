# Nuovo piano

### This is what I was thinking to do https://huggingface.co/spaces/ajitrajasekharan/Image-Text-Detection to recognize text

- [This works good already just as an example](https://huggingface.co/spaces/deepdoctection/deepdoctection) so I think Object Detection is not a problem.

- OCR/HTR is a problem on the other hand, I can try paddleOCR. Also follow the original idea of Fine-tuning TrOCR or perhaps take a look at:

Have a look at Calamari (https://github.com/Calamari-OCR/calamari) and Kraken (https://github.com/mittagessen/kraken)
The best model for historical documents is https://github.com/DCGM/pero-ocr (it used to be hell to integrate, it seems they have improved that though).
For HTR have a look at https://github.com/omni-us/research-seq2seq-HTR.Also build a custom solution perhaps and maybe even use multiple ones.

- This is a good starting point https://portal.vision.cognitive.azure.com/demo/extract-text-from-images. I would like to find something open source though, I should also look at [florance](https://arxiv.org/pdf/2111.11432.pdf) or other open source alternatives.

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


#### First test accuracy of 83% on test set if I remember correctly

The image 005294.jpg was wierd

- Maybe use data augmentation flipping the images and wathsoever

- Perhaps you can also think about generation fake images with handwritings for the labels you can use a model crop the labels and make fake version of the text like in the Donut paper

- Perhaps we can think of giving the model negative examples which can be images without labels but the ground truth must be without labels to

- Possibile do some selfe supervision with a good modle to match plants whicwh names are in the possible names on a larger dataset and then use those as training data

- Learn more about https://github.com/jessevig/bertviz

- Visualize model attention https://github.com/clovaai/donut/issues/45

- Use the synthdog with a new text file names for the species that might be interesting and for the localites to, to teach the model how to read, and then do the finetuning. I think it would be better if the sytnetic generated examples where in a form similar to an herbarium image (which allows us to add a bias which is really useful)

- Reaed this paper on OCR: https://arxiv.org/pdf/2111.02622.pdf

- We can also think of a way to use the unlabeled data and also to have more biases to be able to learn faster

### Things I have to do

_To improve the performance of your model, you can also consider data augmentation techniques, such as adding noise, rotation, or scaling to your labeled examples, to increase the diversity of your dataset and make the model more robust._

### Other resources:

- [Multidimensional Recurrent Layers needed ?](https://ieeexplore.ieee.org/document/8269951)

- I have to take a look at the outliers in the dataset. You can simply run the model on the entire dataset and see the wrong predictions to take a look at those and perhaps exclude them for the next training.

- We really need to have a clean dataset for the model evaluation so I need to clean the dataset

#### What did I do ?

- Command I used to extract sudo find . -type f -exec mv -f {} ../full_images \;



### TODO

- Take a look at this: ```https://huggingface.co/docs/transformers/transformers_agents```

- do something like this https://www.youtube.com/watch?v=71EOM5__vkI and use ocr

- Take a look at [Florence](https://arxiv.org/pdf/2111.11432.pdf)

- Interesting things: https://medium.com/@surve790_52343/transformer-for-ocr-donut-trocr-1a138e9f2cb9

- Maybe use syndog for your custom dataset and tell it how to read and then you can use langchain

- Langchain + LLM + OCR or Trocr like I wanted + lookup on data + internet
