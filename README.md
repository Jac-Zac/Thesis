# HTR on museum's biological artifacts labels

This repository will have a bunch of information for know.

#### TODO:
> Ask chatGPT to write a better paper for you

- [ ] Format the repository
- [ ] Read the previous paper
- [ ] Take inspiration from [this](https://github.com/AlbertoPresta/Thesis)


#### Keywords:

- HTR
- Neural Netowrk
- Machine Learning
- Classification

#### Possible things to do

- Look at the [visual transformer](https://www.youtube.com/watch?v=TrdevFK_am4)
- Take a look at Attention is all you need again


### Response:

```
HTR is a little bit of a dark art as its performance is highly dependent on data quality which depends both on age, writing style, writing substrate, etc...
Broadly there are two worlds:
1. HTR without a language model
2. HTR with a language model
The advantage of having a language model is that corrections that would ordinarily exceed the signal-to-noise ration of the original source can still be fixed by e.g. considering the sentence's grammar.
However, this is also a double edged sword since these models are also very prone to hallucinating or adding in words not in the original source with the frequency of such hallucinations being very well correlated with the strength of the language model.
```

The best thing to do is probably just testing out different models (for example have a look at the work done e.g. [here](https://link.springer.com/chapter/10.1007/978-3-031-06555-2_27).
Also keep in mind that combining multiple predictors can be a very powerful technique as well [see](https://dl.gi.de/handle/20.500.12116/16993)

```
The fact that you have data usually bodes well since this helps a lot to narrow down the specificities of your dataset.
The difference between success and failure is usually directly related in how much noise you can get rid of during preprocessing, so if you can clean you data through e.g. binarization and morphological operators this can make a huge difference.
If you have a rough idea about what is in your dataset, you can also use that to fix your recognitions (e.g. instead of hoping a language model will fix errors, you can do a dictonary lookup for similar terms and potentially fix the results).
```
