# Piano per il progetto

- Restart during summer, read the paper you have to read and do some training test

- read this: [domain adaption](https://towardsdatascience.com/understanding-domain-adaptation-5baa723ac71f)

- A new good Idea could be to have a tranformer trained like a swin traformer and then finetuned by me to recognize the labels to then have something like bing recognize what is written with access to internet usinig an llm.

- Bing likes to stop responding by I could do visual question answering with Donut also think about it. O do something else to. Also enable internet access maybe with langchain ?

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


#### New ideas

Yes, it's possible to fine-tune Donut's visual question answering capabilities on your own dataset and give it access to the internet to respond. Here's a general outline of the steps involved:

    Prepare your dataset: Collect and annotate a dataset of images and their corresponding questions and answers. Make sure the annotations are in a format that Donut can understand.
    Fine-tune Donut on your dataset: Use Donut's built-in tools or libraries like Hugging Face's Transformers to fine-tune the model on your dataset. This step will update the model's weights to better suit your specific use case.
    Give Donut access to the internet: To enable Donut to respond to questions by accessing the internet, you'll need to set up an environment that allows it to send HTTP requests and receive responses. One way to do this is by using a library like requests in Python.
    Modify Donut's response generation code: Once Donut has accessed the internet and retrieved relevant information, you'll need to modify its response generation code to incorporate the new information. This might involve injecting the new information into the model's output or using a different decoder to generate the final response.
    Test and evaluate the updated Donut model: After making changes to the model, test it thoroughly to ensure it's working correctly and providing accurate responses. Evaluate the model's performance on a validation set to assess its effectiveness.
    Deploy the updated Donut model: Once you're satisfied with the model's performance, deploy it so that it can respond to user queries. You can host the model on a server, containerize it, or use a cloud service like AWS Lambda.
    Monitor and maintain the model: As time passes, the model may become outdated or less effective due to changes in the internet landscape or user behavior. Regularly monitor the model's performance and retrain or update it as needed to maintain its accuracy and relevance.

Keep in mind that modifying Donut's architecture or training procedure requires a good understanding of deep learning and natural language processing concepts. It's essential to have experience with PyTorch or another deep learning framework, as well as familiarity with transformer-based architectures like BERT and ResNet.

Additionally, consider the ethical implications of giving Donut access to the internet and allowing it to provide open-ended responses. Ensure that the model is designed with safeguards to prevent misuse or abuse, such as filtering profanity or hate speech, and that users are aware of the potential risks associated with interacting with AI systems.
