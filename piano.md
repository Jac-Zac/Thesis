# Prima cosa da fare dal dataset prendere il testo

Ho letto questa Systematic Literature Review e notata che a pagina 23, 24 parlano di quello che ci interessa. (https://arxiv.org/ftp/arxiv/papers/2104/2104.08732.pdf)

In particolare questo articolo sembra anche interessante: (https://riojournal.com/article/55789/element/4/5731002//)

E invece prima bisogna selezionare il testo per farlo si può inizialmente usare: https://www.researchgate.net/publication/340039970_Objects_Detection_from_Digitized_Herbarium_Specimen_based_on_Improved_YOLO_V3 (anceh se secondo me si può fare di meglio anche qui ma per adesso mi basta YOLO anche normale)

Questo spiega un framework generale: https://docs.google.com/viewerng/viewer?url=https://digital.csic.es/bitstream/10261/239620/1/814319.pdf

A questo punto comunque prenderei un grande dataset e comincerai su quello a fare transfer learning da: https://huggingface.co/docs/transformers/model_doc/trocr

E poi fare fine tuning sul nostro dataset

Vedere quello che si ottiene e poi fare (NER) con un LLM. In caso questo si puo unire a TrOCR in futuro o almeno questo sarebbe quello che vorrei
