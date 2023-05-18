# HTR on biological artifacts labels

### Most Relevant material to read

<!-- ### [`MUST READ by the teacher`](https://direct.mit.edu/dint/article/4/2/320/109837/The-Specimen-Data-Refinery-A-Canonical-Workflow) -->

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
- [X] Re reed Attention is all you need again
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

### Things I have to do

_To improve the performance of your model, you can also consider data augmentation techniques, such as adding noise, rotation, or scaling to your labeled examples, to increase the diversity of your dataset and make the model more robust._

### Other resources:

- [Multidimensional Recurrent Layers needed ?](https://ieeexplore.ieee.org/document/8269951)

- I have to take a look at the outliers in the dataset. You can simply run the model on the entire dataset and see the wrong predictions to take a look at those and perhaps exclude them for the next training.

- We really need to have a clean dataset for the model evaluation

- Also using the median for example gives a way better score which should tell us a lot

- Also high res images seem to give better result but do not use early stopping I still improvement at the 10k example steps and after
