# HTR on biological artifacts labels

### Most Relevant material to read

### [`MUST READ`](https://direct.mit.edu/dint/article/4/2/320/109837/The-Specimen-Data-Refinery-A-Canonical-Workflow)


#### [Comprensive paper](https://arxiv.org/ftp/arxiv/papers/2104/2104.08732.pdf)

#### [`Comprensive Blog good started`](https://nanonets.com/blog/handwritten-character-recognition/)

- [ ] Cool [OCR-transformer](https://arxiv.org/abs/2109.10282) I want to use this and I have to find the pre-trained model to use

- [Convolve Attend and Spell paper](https://priba.github.io/assets/publi/conf/2018_GCPR_LKang.pdf)

- [Pay Attention to What You Read paper](https://arxiv.org/abs/2005.13044)

- [Comparison of Open-Source Libraries paper](https://teklia.com/publications/DAS2022_HUMU.pdf)

- [HTR-Flor paper](https://ieeexplore.ieee.org/document/9266005)

- [Handwriting Recognition of Historical Documents with few labeled dat](https://arxiv.org/pdf/1811.07768v1.pdf)
    > READ dataset ofline HTR

    > [code](https://github.com/0x454447415244/HandwritingRecognitionSystem)

- Potentially ScrabbleGAN to have more data

- Connectionist Temporal Classification (CTC)
    > CTC is a type of neural network architecture that can be used to learn mappings between input (images for example) sequences and output sequences (labels). In speech recognition, for example, the input sequence is an audio signal, and the output sequence is a sequence of words that the speaker is saying.

    > Beam search decoding

#### TODO:
> Ask chatGPT to write a better paper for you

- [ ] Format the repository
- Take a look at [this facebook github repo](https://github.com/facebookresearch/SparseConvNet)
- [ ] Take a look at Attention is all you need again
- [ ] Take inspiration from [this](https://github.com/AlbertoPresta/Thesis)
- [ ] ViT pytorch implementation [hear](https://github.com/lucidrains/vit-pytorch)
- [ ] Read [this](https://nanonets.com/blog/handwritten-character-recognition/) article for information, interesting part start from `Scan, Attend and Read`
- [ ] Read [this](https://paperswithcode.com/task/handwriting-recognition) paper
- [ ] [`The Illustrated Transformer`](https://jalammar.github.io/illustrated-transformer/)

- [ ] [`Seq2seq with attention`](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

- [X] Read the paper they gave me on CV (Computer Vision)

#### Possible things to do

- Transfer-learning from previous work
- Look at the [visual transformer](https://www.youtube.com/watch?v=TrdevFK_am4)
- Distillation in ViTs [hear](https://arxiv.org/abs/2012.12877) but not really that interested for know
- [Paper summary on visual transformer](https://arxiv.org/abs/2012.12556)

- [Small ViT](https://arxiv.org/abs/2106.10270)

#### Important link

Also keep in mind that combining multiple predictors can be a very powerful technique as well [see](https://dl.gi.de/handle/20.500.12116/16993)

### Response:

```
HTR is a little bit of a dark art as its performance is highly dependent on data quality which depends both on age, writing style, writing substrate, etc...
Broadly there are two worlds:
1. HTR without a language model
2. HTR with a language model
The advantage of having a language model is that corrections that would ordinarily exceed the signal-to-noise ration of the original source can still be fixed by e.g. considering the sentence's grammar.
However, this is also a double edged sword since these models are also very prone to hallucinating or adding in words not in the original source with the frequency of such hallucinations being very well correlated with the strength of the language model.

The fact that you have data usually bodes well since this helps a lot to narrow down the specificities of your dataset.
The difference between success and failure is usually directly related in how much noise you can get rid of during preprocessing, so if you can clean you data through e.g. binarization and morphological operators this can make a huge difference.
If you have a rough idea about what is in your dataset, you can also use that to fix your recognitions (e.g. instead of hoping a language model will fix errors, you can do a dictonary lookup for similar terms and potentially fix the results).
```

### Things I have to do

_To improve the performance of your model, you can also consider data augmentation techniques, such as adding noise, rotation, or scaling to your labeled examples, to increase the diversity of your dataset and make the model more robust._

### Other resources:

- [transkribus github](https://github.com/transkribus/)

- PyLaia and Kaldi

- [Multidimensional Recurrent Layers needed ?](https://ieeexplore.ieee.org/document/8269951)

### Dataset:

- *SERNEC* really good
- IDigBio
<!-- - GBIF -->

> Pheraps you trasfer learn on thouse and then do fine tuning on your dataset

- herbarium challenge 2019 dataset and the PlantCLEF202

- https://doi.org/10.1080/23818107.2018.1446357

- Github repo https://github.com/Ab-Abdurrahman/ubd-herbarium-repository

107] A. Barber, D. Lafferty, L.R. Landrum, The SALIX method: A semi-automated workflow for herbarium specimen

digitization, Taxon. 62 (2013) 581–590. https://doi.org/10.12705/623.16.
[108] P.B. Heidorn, Q. Zhang, Label Annotation through Biodiversity Enhanced Learning, in: IConference 2013 Proc., 2013:

pp. 882–884. https://doi.org/10.9776/13450.

[109] E.R. Ellwood, B.A. Dunckel, P. Flemons, R. Guralnick, G. Nelson, G. Newman, S. Newman, D. Paul, G. Riccardi, N.

Rios, K.C. Seltmann, A.R. Mast, Accelerating the digitization of biodiversity research specimens through online
public participation, Bioscience. 65 (2015) 383–396. https://doi.org/10.1093/biosci/biv005.

#### Remember

Use batchnorm to control the statistic of activation we place it after Conv or linear Layers, no need for bias in the previous layer if you use it
