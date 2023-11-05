# Solution progress report
This report contains an overview of the path to the final solution.

## Initial idea
Initial idea was to sligtly adapt lab notebook, and fine-tune small T5 model to transform toxic text sentences into neutral ones by providing model with reference texts and their neutral tranlsations, contained in the dataset. 

## Changes along the way

### Prefix
I intended to use a prompt/prefix added to each toxic sentence before feeding it into the model like: `detoxify:<sentence_to_detoxify>`. After a couple of tests and addressing T5 model docs, it was decided to not use any prefix, since it sligtly shortens the sentences, and doesn't seem to affect final results.

### Generation limit
Generation limit does not seem to have a significat effect on training performance, so it was chosen to set it to 64, since over 99.9% of the dataset sentences are 64 tokens or less.

### Training parameters
I was primarily experementing with learning rate, since every time I changed it the results also changed significantly. For the final model I tried:
- lr = 5e-5, weight_decay = 0.04
    - about 58% successfully detoxified samples from validation set
- lr = 1e-4, weight_decay = 0.01
    - about 60.5% successfully detoxified samples from validation set
- lr = 5e-4, weight_decay = 0.01
    - about 67% successfully detoxified samples from validation set
- lr = 1e-3, weight_decay = 0.01
    - about 68.5% successfully detoxified samples from validation set

### Training process
According to the paper on text detoxification, there are no standard and efficient metrics for detoxification seq2seq training, therefore a loss calculated by the T5 model was used for training.

### Data
Initially I was using only simple data preprocessing: sorting references and translations
to ensure model always gets toxic input and neutral label.

However, the final model was trained on the cleaned version of the dataset. I used my T5-based toxicity regressor (more on it down below) to reevaluate toxicity scores of the given samples. After reevaluation, rows with both samples evaluated as neutral or toxic, as well as rows that were evaluated as neutral -> toxic, were removed from the dataset.

## Model evalutation
The initial plan was to train a transformer-based regressor that for evaluating the toxicity of text. The idea was to take an encoder from T5 small model, add a head for regression, and train the resulting model on the same dataset, inputs being reference/translation texts, and the labels are their respective toxicity levels.

After that the toxicity threshold can be chosen in order to analyze, how many toxic sentences were successfully translated into a neutral ones.

### Regressor architecture
Initial regressor architecture was T5 encoder followed by 3-layerd fully-connected NN. The way the toxicity score is calculated is by passing hidden state returned by encoder for each token, and then sum up the results into a single number, activated by a sigmoid function to produce an output number from 0 to 1. 

The final architecture is essentially the same, but instead of 3 layers, there is only 1 FC layer, since the amount of layers didn't seem to affect the model's perfomance.

### RoBERTa toxicity classifier

Later I found out about existing toxicity classifier model made by the authors of the beforementioned paper. This is a RoBERTa based classifier, which, given a sentence, decides, whether it belongs to class `neutral` vs class `toxic`. This model could also be used to evalutae detoxification model's perfomance.

I ended up training my own regressor anyway, after which I compared performance of both my regressor and RoBERTa-based classifier. In general models seemed to have rather similar performance: there was about 15% discrepancy between the dataset labels and models' predictions, which is arguably expected given the quality of the data and knowing that the term `toxicity` is vague, which further adds subjectiveness of given toxicity estimations. Inspection of examples of sentences, classified as toxic vs. netural by both models shown that both models do a decent job at detecting toxicity. However, my regressor seemed to be give results more similar to the ones presented in the dataset. Notably, this does not mean that my model is a superior toxicity classifier, but rather in is better at detecting toxic text which is generally considered toxic in terms of this specific dataset.

Finally I decided to use my regressor to evaluate model's performance, but I believe that the results would be quite similar were I to use the RoBERTa based model instead.

## Results

My final solution is a fine-tuned T5 small model, since it's the only one I tried (since it had arguably decent performance). As for my training arguments, my best results were by using $\text{learning\_rate} = 0.001, \text{weight\_decay} = 0.01$. I got better results when training on the cleaned version of the dataset, however whether cleaning the dataset how I did is valid is not entirely clear for me.

As for evalutaion models, I decided to use both T5 regressor and RoBERTa classifier and compare their results. More evaluations - more info