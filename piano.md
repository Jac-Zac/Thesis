# Prima cosa da fare dal dataset prendere il testo

- [X] Ho letto questa [Systematic Literature Review](https://arxiv.org/ftp/arxiv/papers/2104/2104.08732.pdf) e notata che a pagina 23, 24 parlano di quello che ci interessa.

- [ ] In particolare questo articolo sembra anche interessante: [Towards a scientific workflow + NLP](https://riojournal.com/article/55789/element/4/5731002//)

- [X] E invece prima bisogna selezionare il testo per farlo si può inizialmente usare: https://www.researchgate.net/publication/340039970_Objects_Detection_from_Digitized_Herbarium_Specimen_based_on_Improved_YOLO_V3 (anceh se secondo me si può fare di meglio anche qui ma per adesso mi basta YOLO anche normale)

- [ ] Questo spiega un framework generale: https://docs.google.com/viewerng/viewer?url=https://digital.csic.es/bitstream/10261/239620/1/814319.pdf


A questo punto comunque prenderei un grande dataset e comincerai su quello a fare transfer learning a partire da [TrOCR](https://huggingface.co/docs/transformers/model_doc/trocr)

> Magari per il transfer learning aggiungere dei token speciali tipo dei CLS (classification token) ovviamente voglio usare Transformers e per ocr ViT

E poi fare fine tuning sul nostro dataset

Vedere quello che si ottiene e poi fare (NER) con un LLM. In caso questo si può unire a TrOCR in futuro o almeno questo sarebbe quello che vorrei per fare end to end.

#### Problem statement:

```
I have images from an herbarium, I need to retrive the data from the labels (which can contain species name, name of the author, date, etc...) I was thinking of using something like YOLO to find the label and crop only those from my entire dataset and then use TrOCR to recognize the text from every image and then do (NER) to devide them in the correct entities. I need your help first of all what do you think about this plan is it feasable ?```

```
Oky I have a problem though TrOCR need a line of text If I give it the entire label I don't think It will work so what can I do ? Can I just go with that and it will learn to do it in the transfer learning stage ? I do not think so so what should I do ? Can I perhaps devide the labels in multiples line somehow automatically ? Because if I try that I think I need to recognize the lines of text and I do not know how to do that. Maybe there is something that can help me with that
```


#### Take a look at [this for recognition of text](https://arxiv.org/abs/2104.07787)
