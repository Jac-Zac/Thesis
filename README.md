# HTR on biological artifacts labels

## Summary of my work 📝

#### Pipline approach

- Firstly I started to take a look at how to subdivide the recognition with a pipeline. A template of the idea of my pipline that I made after can be [in this streamlit app](streamlit_test/herbarium.py)

-

#### End to end approach
> This is the main approach I have explored for now since it required less time to get running (and also I have an interested in Multimodal models)

##### Document Understanding for my problem

- The main idea is to use a model that can do document understanding and make it suitable for our use case.

Btw I also though of training 1 or more models from scratch, which considering the "relatively small amount of data and compute" can't be something like a tranformer but should be something with bigger implicit biases, to facilitate the training since in general things like CNN for example have better accuracy for small dataset because of the intrinsic transational invariance bias  (with max pooling) and the fact that they have local dipendencyes, which is something that more general ViT do not have. Also in our case for the Donut model the encoder is a Swin Transformer which ideally would make sense to work better then a standard ViT because of the fact that it allows to attentd also inside the patches (this is a general explainantion of why I think it is a good idea). Then the bart model inside donut helps for the NER.

- I though about trying to investigate into the model firstly by taking a look at the failure cases [hear](link) and in the future also investigate the attentoin heatmap perhaps

##### What I have done as of know

Weights and biases [draft](https://api.wandb.ai/links/jac-zac/3uf34i1s)

Fine tuned some version of the base Donut model that can be found on hugginface by adding some new tokens and a task token (everything aoubt it can be found in the Donut finetuning notebook that I modified from the original one provided on hugginface). And I did the finetning on around 1.5k images with train, validation and test split obviusly taking results with weights and biases. Though I plan to do it better and track more metric and also do some hyperparameters search to find the best ones. To do that I downscaled the images to fit into the GPU I had available in Kaggle though increasing the resolution a bit seemd to give better resutls this is why in the final training I'd ideally work with images of at least around 2400 x 1800 which is still a significant dowscaled version of the original images which can vary from 2 or 4 times this resolution.

##### Future ideas

- I'll create a new huggingface directory and start to do some real test with the full dataset

- Perhaps also have a mixture of models I have to study more on other things also such as soft mixture of expert (even though this is what you do when you are out of idea ... GPT4 ...)

- Perhaps we can connect it to a vector database or do some similarty search on the species names to see if they match the ones provided for possibel species. (To see in the future)

### Papers about the HTR process that might be helpful

- [Convolve Attend and Spell paper](https://priba.github.io/assets/publi/conf/2018_GCPR_LKang.pdf)

- [Handwriting Recognition of Historical Documents with few labeled dat](https://arxiv.org/pdf/1811.07768v1.pdf)

- [Pay Attention to What You Read paper](https://arxiv.org/abs/2005.13044)

- [HTR-Flor paper](https://ieeexplore.ieee.org/document/9266005)

    > [code](https://github.com/0x454447415244/HandwritingRecognitionSystem)

- Potentially ScrabbleGAN to have more data


### Other resources

[Donut really good](https://huggingface.co/docs/transformers/main/en/model_doc/donut)

Interesting OCR/HTR. [I have to watch](https://www.youtube.com/watch?v=8VLkaf_hGdQ)


#### TODO for me:

- [ ] Format the repository
- [ ] Read [this](https://nanonets.com/blog/handwritten-character-recognition/) article for information, interesting part start from `Scan, Attend and Read`
- [ ] Read [this](https://paperswithcode.com/task/handwriting-recognition) paper
- [ ] [`The Illustrated Transformer`](https://jalammar.github.io/illustrated-transformer/)
- [ ] [`Seq2seq with attention`](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

#### Important link

Also keep in mind that combining multiple predictors can be a very powerful technique as well [see](https://dl.gi.de/handle/20.500.12116/16993)
