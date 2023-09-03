# HTR on biological artifacts labels 🌱

## Summary of my work 📝

### Pipeline approach

My original idea was to approach the problem by crating a pipeline constructed of:
> Take a look at [this](https://mindee.github.io/doctr/)

- Text detection _(which I do not think should be too hard)_.

- OCR/HTR. We could use many different models, my initial idea was to fine tune [TrOCR](https://arxiv.org/pdf/2109.10282.pdf)

- NER (Name Entity Recognition). Simply use an LLM, honestly I think any decent LORA 7b model would pretty much do the job.

- Compare the species names with the possible ones (maybe we will have a file for it).

> The template code for my idea can be found [here](https://github.com/Jac-Zac/Thesis/blob/master/pipline_example/herbarium_app.py) even though some pieces are missing.

### End to end approach
> This is the main approach I have explored as of now, since it required less time to get running.

#### Leveraging advances in Document Understanding for my problem

The main idea is to use a model that can do document understanding and make it suitable for our use case. The reason why I didn't developed a model from scratch will be explained later. (Even though I perhaps will explore the idea of custom models in the future).

- I have looked at a few different models for Document Understanding and found [Donut](https://arxiv.org/pdf/2111.15664.pdf)

- I have created a [few notebook](https://github.com/Jac-Zac/Thesis/tree/master/Donut_notebooks) to get the model running on my custom dataste.

- After a few runs that were uploaded to [this repo](https://huggingface.co/Jac-Zac/thesis_test_donut).

- I have also started looking at [failure cases here](https://github.com/Jac-Zac/Thesis/blob/5bd9f8c58216e776efb6cc57b0b09665bd20a99d/inference/model_evaluation.ipynb). In the future I also want to investigate the cross attention heatmap by modifying [this code](https://github.com/Jac-Zac/Thesis/blob/master/inference/template_for_cross_attention_heatmap_and_bounding_box.ipynb). Refer to [this discussion](https://github.com/clovaai/donut/issues/45) for more info about it.

### My reasoning

I considered training one or more custom models from scratch, taking into account the "relatively small amount of data and compute" available. Instead of using a transformer architecture, I would have opted for an alternative architecture with stronger implicit biases to facilitate training. CNNs generally exhibit better accuracy on small datasets due to their intrinsic biases, such as:

- Translation invariance (resulting from the combination of convolution and max-pooling)

- Local sensitivity

In our case, the Donut model which has a Swin Transformer as its encoder, should theoretically outperform a standard ViT which only has simple attention across patches. Furthermore, the BART model used as a decoder should aid in Named Entity Recognition (NER)

### What I have done as of know

I fine-tuned a version of the base [Donut model available on Hugging Face](https://huggingface.co/docs/transformers/model_doc/donut), by adding new tokens and a task token.

Details about this process can be found in the modified Donut fine-tuning notebook, which I adapted from the original one provided on [here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_(CORD)_with_PyTorch_Lightning.ipynb)

- I fine-tuned the model on approximately 1.5k images, with train, validation, and test splits, and tracked the results using [Weights & Biases](https://wandb.ai)
- In the future, I plan to track more metrics and perform hyperparameter searches to find the best ones.
- To fit the images into the GPU available on Kaggle, I downscaled them. However, increasing the resolution slightly seemed to yield better results. Therefore, in the final training, I would ideally work with images of at least around 2400 x 1800 pixels. This is still a significantly downscaled version of the original images, which can vary from 2 to 4 times this resolution.

The initial runs where recorded on Weights and biases and [this is a draft of an initial report](https://api.wandb.ai/links/jac-zac/3uf34i1s)

### Future ideas

- Train on [Lambda](https://lambdalabs.com/) (probably on an A100 or H100)

- I will create a new Hugging Face directory and begin testing with the full dataset.

- I may consider using a mixture of models and explore other techniques, such as soft mixtures of experts. Or even just have two different approach and do something with it to get better performance.

- We could potentially connect the model to a vector database or perform similarity searches on species names to check for matches with provided species names (to be explored in the future).

- Potentially generate fake data for training also with possible names. Also data augmentation technique.

### Papers about the HTR process that might be helpful

- [Convolve Attend and Spell paper](https://priba.github.io/assets/publi/conf/2018_GCPR_LKang.pdf)

- [Handwriting Recognition of Historical Documents with few labeled data](https://arxiv.org/pdf/1811.07768v1.pdf)

- [Pay Attention to What You Read paper](https://arxiv.org/abs/2005.13044)

- [HTR-Flor paper](https://ieeexplore.ieee.org/document/9266005)

    > [code](https://github.com/0x454447415244/HandwritingRecognitionSystem)

## TODO (for myself):

- Big problems with unreliable dataset (needs discussion)
- I have to take a look at the shape the images are passed to the model

- [X] Format the repository
- [X] New model first epoch, took a lot of compute but quite good on unseen data
- [ ] Understand why some images get no prediction and when cropped even slightly they get the exact prediction. Maybe look into data augmentation.
- [ ] Look into the model auto alleging to the width automatically
- [ ] The model is already ready for multiple ground truth though we should add them to the csv file if we want to do that.
- [ ] Format the csv file better
- [ ] Kaggle train from checkpoint to get the first epoch
- [ ] To read [this paper for new model Document Understanding](https://arxiv.org/pdf/2307.02499.pdf)
- [ ] Read [this](https://nanonets.com/blog/handwritten-character-recognition/) article for information, interesting part start from `Scan, Attend and Read`
- [ ] Read [this](https://paperswithcode.com/task/handwriting-recognition) paper
- [ ] Take a look at this again: [`Seq2seq with attention`](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [ ] Interesting OCR/HTR. [I have to watch](https://www.youtube.com/watch?v=8VLkaf_hGdQ)

- Also keep in mind that combining multiple predictors can be a very powerful technique as well [see](https://dl.gi.de/handle/20.500.12116/16993)
