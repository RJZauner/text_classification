<a href="https://colab.research.google.com/github/RJZauner/text_classification/blob/main/hf_distilbert_text_classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# NLP Basics: Text Classification with Hugging Face

I have always found that when learning some new concept, I like to get typing as soon as possible.

I want to experiment with what I know to learn more about concepts, even if I think I know how something works - you never know what you might learn along the way.

With machine learning and natural language processing (NLP) it is no different.

Let us now have a look at a "Hello World" project for NLP: **Text Classification**.

## What we will learn
- Binary (2 Categories: Positive and Negative) Text Classification
- Training a Pre-Trained Model (Distilbert uncased)
- Using Hugging Face Dataset to load the Rotten Tomatoes Dataset
- Hosting model on the Hugging Face hub

## What is Text Classificaton

The idea behind text classification is really simple.

Based on text that a person has written or spoken, the model tries to find patterns for certain pre-defined categories:

"This product solved all of our issues - cannot recommend enough!"

"Would definitely **not** recommend under any circumstances"

As you can tell, the first sentence would be a positive review of a product, while the second sentence is clearly more negative.

This is an example of binary classification, where the model places the given text (e.g. "This product solved all of our issues - cannot recommend enough!") in one of the two categories (in this case "Positive").

A text classification task can have any number of cateogories and is not restricted to just two.

The model uses certain patterns in sentences to identify the category a sentence belongs to.

We do pretty much the same thing. By containing certain words in certain places, a sentence can have a more positive meaning or a more negative meaning.

Our model simply takes in the text and calculates how likely it is, that the given sentence belongs to a positive category or, how likely it is, that the sentence bleongs to a negative category.

In the end it's all numbers - after all, that is what computers understand.

Now that we have a high-level overview,  we can start with everyone's favourite: **Data**.

## Data
For this task we will be using the rotten_tomates dataset which contains positive and negative reviews.

This is therefore a binary classification task, seeing as our model has to predict 2 categories: Positive and Negative.

In a real-world project, this is usually the point where we need to evaluate how much data we have and if we have the right data in the right quality.

The focus of our project here is to learn more about text classification and as such I will be making use of Hugging Face's dataset library  to load in the rotten tomatoes dataset.

## Installation

Let us first get the installation out of the way.

Here we will install datasets, transformers and sentencepiece for our development.

The huggingface_hub library is not necessary for development. I included it to make sharing the model easier.

If you are following along, feel free to comment that line out.



```python
! pip install datasets
! pip install huggingface_hub
! pip install transformers
! pip install sentencepiece
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting datasets
      Downloading datasets-2.4.0-py3-none-any.whl (365 kB)
    [K     |████████████████████████████████| 365 kB 28.7 MB/s 
    [?25hRequirement already satisfied: aiohttp in /usr/local/lib/python3.7/dist-packages (from datasets) (3.8.1)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from datasets) (1.21.6)
    Collecting multiprocess
      Downloading multiprocess-0.70.13-py37-none-any.whl (115 kB)
    [K     |████████████████████████████████| 115 kB 70.8 MB/s 
    [?25hRequirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from datasets) (21.3)
    Collecting responses<0.19
      Downloading responses-0.18.0-py3-none-any.whl (38 kB)
    Collecting xxhash
      Downloading xxhash-3.0.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (212 kB)
    [K     |████████████████████████████████| 212 kB 72.7 MB/s 
    [?25hCollecting huggingface-hub<1.0.0,>=0.1.0
      Downloading huggingface_hub-0.9.0-py3-none-any.whl (120 kB)
    [K     |████████████████████████████████| 120 kB 74.2 MB/s 
    [?25hRequirement already satisfied: dill<0.3.6 in /usr/local/lib/python3.7/dist-packages (from datasets) (0.3.5.1)
    Requirement already satisfied: fsspec[http]>=2021.11.1 in /usr/local/lib/python3.7/dist-packages (from datasets) (2022.7.1)
    Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from datasets) (1.3.5)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (2.23.0)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from datasets) (4.12.0)
    Requirement already satisfied: pyarrow>=6.0.0 in /usr/local/lib/python3.7/dist-packages (from datasets) (6.0.1)
    Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.7/dist-packages (from datasets) (4.64.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (4.1.1)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (6.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets) (3.8.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->datasets) (3.0.9)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (2022.6.15)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (2.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.19.0->datasets) (1.24.3)
    Collecting urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1
      Downloading urllib3-1.25.11-py2.py3-none-any.whl (127 kB)
    [K     |████████████████████████████████| 127 kB 73.1 MB/s 
    [?25hRequirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.2.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (6.0.2)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (1.8.1)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (22.1.0)
    Requirement already satisfied: asynctest==0.13.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (0.13.0)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (4.0.2)
    Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->datasets) (2.1.0)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->datasets) (3.8.1)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas->datasets) (2022.2.1)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas->datasets) (1.15.0)
    Installing collected packages: urllib3, xxhash, responses, multiprocess, huggingface-hub, datasets
      Attempting uninstall: urllib3
        Found existing installation: urllib3 1.24.3
        Uninstalling urllib3-1.24.3:
          Successfully uninstalled urllib3-1.24.3
    Successfully installed datasets-2.4.0 huggingface-hub-0.9.0 multiprocess-0.70.13 responses-0.18.0 urllib3-1.25.11 xxhash-3.0.0
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting transformers
      Downloading transformers-4.21.1-py3-none-any.whl (4.7 MB)
    [K     |████████████████████████████████| 4.7 MB 8.7 MB/s 
    [?25hRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (1.21.6)
    Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers) (3.8.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.7/dist-packages (from transformers) (6.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers) (2.23.0)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.7/dist-packages (from transformers) (4.64.0)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from transformers) (4.12.0)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (2022.6.2)
    Collecting tokenizers!=0.11.3,<0.13,>=0.11.1
      Downloading tokenizers-0.12.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.6 MB)
    [K     |████████████████████████████████| 6.6 MB 39.0 MB/s 
    [?25hRequirement already satisfied: huggingface-hub<1.0,>=0.1.0 in /usr/local/lib/python3.7/dist-packages (from transformers) (0.9.0)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from transformers) (21.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0,>=0.1.0->transformers) (4.1.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->transformers) (3.0.9)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->transformers) (3.8.1)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (1.25.11)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2022.6.15)
    Installing collected packages: tokenizers, transformers
    Successfully installed tokenizers-0.12.1 transformers-4.21.1
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting sentencepiece
      Downloading sentencepiece-0.1.97-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    [K     |████████████████████████████████| 1.3 MB 31.1 MB/s 
    [?25hInstalling collected packages: sentencepiece
    Successfully installed sentencepiece-0.1.97



```python
from datasets import load_dataset_builder, load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from google.colab import output
from huggingface_hub import notebook_login
import numpy as np

output.enable_custom_widget_manager() #enable third-party-widgets
```


```python
notebook_login()
```

### Inspecting our Dataset

You may be building this on a cloud-service provider, where memory is critical.

As such, you can't simply download every dataset and store them on your session - you may run out of memory.

Hugging Face provides us a really simple way of inspecting datasets that are stored in the Hugging Face Hub using the `load_dataset_builder("<name_of_dataset_goes_here>")` method.

This will download the meta-information regarding your dataset, such as the description,  feature  information, citation etc. and provides a simple way of gathering information on your dataset without having to leave your notebook.

Let us create a new `ds_builder` object and get the description, to decide if this dataset suits our needs:


```python
dataset_name = "rotten_tomatoes"
```


```python
# Inspect dataset before committing to downloading
ds_builder = load_dataset_builder(dataset_name)

ds_builder.info.description
```


    Downloading builder script:   0%|          | 0.00/1.89k [00:00<?, ?B/s]



    Downloading metadata:   0%|          | 0.00/921 [00:00<?, ?B/s]


    WARNING:datasets.builder:Using custom data configuration default





    "Movie Review Dataset.\nThis is a dataset of containing 5,331 positive and 5,331 negative processed\nsentences from Rotten Tomatoes movie reviews. This data was first used in Bo\nPang and Lillian Lee, ``Seeing stars: Exploiting class relationships for\nsentiment categorization with respect to rating scales.'', Proceedings of the\nACL, 2005.\n"



Let us now have a peek at the features:


```python
ds_builder.info.features
```




    {'text': Value(dtype='string', id=None),
     'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None)}



Looks good.

A full list of attributes can be found here: https://huggingface.co/docs/datasets/v2.4.0/en/package_reference/main_classes#datasets.DatasetInfo

Now that we sure we want to use this dataset, we can proceed to load it using the other method (`load_dataset("<name_of_dataset_goes_here>")`) we imported earlier.

To do this, we first need to load in our data.


```python
ds_dictionary = load_dataset(dataset_name)
```

    WARNING:datasets.builder:Using custom data configuration default


    Downloading and preparing dataset rotten_tomatoes/default (download: 476.34 KiB, generated: 1.28 MiB, post-processed: Unknown size, total: 1.75 MiB) to /root/.cache/huggingface/datasets/rotten_tomatoes/default/1.0.0/40d411e45a6ce3484deed7cc15b82a53dad9a72aafd9f86f8f227134bec5ca46...



    Downloading data:   0%|          | 0.00/488k [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/8530 [00:00<?, ? examples/s]



    Generating validation split:   0%|          | 0/1066 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/1066 [00:00<?, ? examples/s]


    Dataset rotten_tomatoes downloaded and prepared to /root/.cache/huggingface/datasets/rotten_tomatoes/default/1.0.0/40d411e45a6ce3484deed7cc15b82a53dad9a72aafd9f86f8f227134bec5ca46. Subsequent calls will reuse this data.



      0%|          | 0/3 [00:00<?, ?it/s]


Hugging Face provides an additional argument `split` that allows us to specify which dataset we want to load.

We did not provide this argument and as such, we download the all available datasets.

This will download the dataset into the root directory of your development environment.

Once finished, we can check that the download was successful by printing the `train_ds` dictionary.


```python
ds_dictionary
```




    DatasetDict({
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 8530
        })
        validation: Dataset({
            features: ['text', 'label'],
            num_rows: 1066
        })
        test: Dataset({
            features: ['text', 'label'],
            num_rows: 1066
        })
    })



As part of our dataset, we have train, test and validation.

To get the first row of our training dataset, we can use the key "train" followed by the index for the first entry,  which in Python is a `0`:


```python
ds_dictionary["train"][10]
```




    {'text': 'this is a film well worth seeing , talking and singing heads and all .',
     'label': 1}



As you can see we receives an object with the fields "text" and "label".

The text is our input - the data our model will see in production and that it will categorise.

The label is what we are trying to predict - the label our model needs to match our text to.

In this case we have a "1", which stands for a negative review.

We can also return a subset of rows using slicing.

Let us quickly return the first 3 rows of our training dataset:


```python
ds_dictionary["train"][:3]
```




    {'text': ['the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',
      'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson\'s expanded vision of j . r . r . tolkien\'s middle-earth .',
      'effective but too-tepid biopic'],
     'label': [1, 1, 1]}



### Pre-Processing
Now that we have our data loaded, we can proceed to taking the necessary processing steps to get our data into the right shape.

Specifically, we need to take our text and turn it into a set of numbers. After all, computers are only able to understand numbers. The first will be to run a so-called **Tokenizer** to separate each word or each part of a word into tokens. The type of tokenizer we choose depends on the model we pick for our problem.

In this case we will be looking at `distilbert-base-uncased`.

You may be wondering why we pick the small version and the reason is that in the beginning, we are still experimenting and iterating our solution. We want to the training to go quickly to see if our changes to the pre-processing have the desired effect.

After we confirmed the pre-processing we can move on to larger models and see how they improve performance.




```python
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
```


    Downloading tokenizer_config.json:   0%|          | 0.00/28.0 [00:00<?, ?B/s]



    Downloading config.json:   0%|          | 0.00/483 [00:00<?, ?B/s]



    Downloading vocab.txt:   0%|          | 0.00/226k [00:00<?, ?B/s]



    Downloading tokenizer.json:   0%|          | 0.00/455k [00:00<?, ?B/s]


From the code above we can see that we define our model as a string.

We then use Hugging Face's Autotokenizer to separate each part of the sentence into smaller building blocks.


```python
tokenizer.tokenize("Hello, this text will be tokenized!")
```




    ['hello', ',', 'this', 'text', 'will', 'be', 'token', '##ized', '!']



Asyou can see, the tokenizer separates each words of the sentence into an element.

Longer words such as "tokenizer" are split into two separate tokens, but the information that it is the same word is still encoded through the "##" at the beginning of the second chunk.

It is this list of tokens that gets converted into numbers, which is referred to as **numericalization**. 

To apply this to our entire dataset, we can create a function and then use `.map()` to update each row.


```python
def tokenize_input(ds):
  return tokenizer(ds["text"], truncation = True)
```


```python
tokenized_ds = ds_dictionary.map(tokenize_input, batched = True)
```


      0%|          | 0/9 [00:00<?, ?ba/s]



      0%|          | 0/2 [00:00<?, ?ba/s]



      0%|          | 0/2 [00:00<?, ?ba/s]



```python
tokenized_ds
```




    DatasetDict({
        train: Dataset({
            features: ['text', 'label', 'input_ids', 'attention_mask'],
            num_rows: 8530
        })
        validation: Dataset({
            features: ['text', 'label', 'input_ids', 'attention_mask'],
            num_rows: 1066
        })
        test: Dataset({
            features: ['text', 'label', 'input_ids', 'attention_mask'],
            num_rows: 1066
        })
    })




```python
first_row = tokenized_ds["train"][0]

first_row["text"], first_row["input_ids"]
```




    ('the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',
     [101,
      1996,
      2600,
      2003,
      16036,
      2000,
      2022,
      1996,
      7398,
      2301,
      1005,
      1055,
      2047,
      1000,
      16608,
      1000,
      1998,
      2008,
      2002,
      1005,
      1055,
      2183,
      2000,
      2191,
      1037,
      17624,
      2130,
      3618,
      2084,
      7779,
      29058,
      8625,
      13327,
      1010,
      3744,
      1011,
      18856,
      19513,
      3158,
      5477,
      4168,
      2030,
      7112,
      16562,
      2140,
      1012,
      102])




```python
model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           num_labels = 2)
```


    Downloading pytorch_model.bin:   0%|          | 0.00/256M [00:00<?, ?B/s]


    Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_layer_norm.weight', 'vocab_layer_norm.bias', 'vocab_transform.weight', 'vocab_projector.weight', 'vocab_projector.bias', 'vocab_transform.bias']
    - This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.weight', 'pre_classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
```


```python
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis = -1)
  return metric.compute(predictions = predictions, references = labels)
```


    Downloading builder script:   0%|          | 0.00/1.65k [00:00<?, ?B/s]



```python
lr = 2e-5
batch_size = 64
epochs = 5

args = TrainingArguments(
  output_dir = "outputs",
  learning_rate = lr,
  per_device_train_batch_size = batch_size,
  per_device_eval_batch_size = batch_size,
  num_train_epochs = epochs,
  weight_decay = 0.01,
  evaluation_strategy = "epoch",
  report_to = "none",
  push_to_hub = True
)
```

    PyTorch: setting up devices



```python
trainer = Trainer(
  model = model,
  args = args,
  train_dataset = tokenized_ds["train"],
  eval_dataset = tokenized_ds["test"],
  tokenizer = tokenizer, # Tokenizer is passed again to ensure that each row is padded using the preferences of the model's tokenizer
  data_collator = data_collator,
  compute_metrics = compute_metrics
 )
```

    Cloning https://huggingface.co/RJZauner/outputs into local empty directory.
    WARNING:huggingface_hub.repository:Cloning https://huggingface.co/RJZauner/outputs into local empty directory.



```python
trainer.train()
```

    The following columns in the training set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
    /usr/local/lib/python3.7/dist-packages/transformers/optimization.py:310: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      FutureWarning,
    ***** Running training *****
      Num examples = 8530
      Num Epochs = 5
      Instantaneous batch size per device = 64
      Total train batch size (w. parallel, distributed & accumulation) = 64
      Gradient Accumulation steps = 1
      Total optimization steps = 670




    <div>

      <progress value='670' max='670' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [670/670 03:29, Epoch 5/5]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.594039</td>
      <td>0.833959</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>0.709530</td>
      <td>0.822702</td>
    </tr>
    <tr>
      <td>3</td>
      <td>No log</td>
      <td>0.727622</td>
      <td>0.832083</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.065000</td>
      <td>0.769303</td>
      <td>0.841463</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.065000</td>
      <td>0.792742</td>
      <td>0.838649</td>
    </tr>
  </tbody>
</table><p>


    The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 1066
      Batch size = 64
    The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 1066
      Batch size = 64
    The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 1066
      Batch size = 64
    Saving model checkpoint to outputs/checkpoint-500
    Configuration saved in outputs/checkpoint-500/config.json
    Model weights saved in outputs/checkpoint-500/pytorch_model.bin
    tokenizer config file saved in outputs/checkpoint-500/tokenizer_config.json
    Special tokens file saved in outputs/checkpoint-500/special_tokens_map.json
    tokenizer config file saved in outputs/tokenizer_config.json
    Special tokens file saved in outputs/special_tokens_map.json
    The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 1066
      Batch size = 64
    The following columns in the evaluation set don't have a corresponding argument in `DistilBertForSequenceClassification.forward` and have been ignored: text. If text are not expected by `DistilBertForSequenceClassification.forward`,  you can safely ignore this message.
    ***** Running Evaluation *****
      Num examples = 1066
      Batch size = 64
    
    
    Training completed. Do not forget to share your model on huggingface.co/models =)
    
    





    TrainOutput(global_step=670, training_loss=0.05473052209882594, metrics={'train_runtime': 210.0501, 'train_samples_per_second': 203.047, 'train_steps_per_second': 3.19, 'total_flos': 618222762273672.0, 'train_loss': 0.05473052209882594, 'epoch': 5.0})



##  Publishing your work

Now that we have a working model, we can push our work to Huggingface Hub. and allow others to join in the open-source fun.


```python
trainer.push_to_hub("distilbert_rotten_tomatoes_text_classifier")
```

    Saving model checkpoint to outputs
    Configuration saved in outputs/config.json
    Model weights saved in outputs/pytorch_model.bin
    tokenizer config file saved in outputs/tokenizer_config.json
    Special tokens file saved in outputs/special_tokens_map.json
    Several commits (2) will be pushed upstream.
    WARNING:huggingface_hub.repository:Several commits (2) will be pushed upstream.
    The progress bars may be unreliable.
    WARNING:huggingface_hub.repository:The progress bars may be unreliable.



    Upload file pytorch_model.bin:   0%|          | 3.34k/255M [00:00<?, ?B/s]


    remote: Scanning LFS files for validity, may be slow...        
    remote: LFS file scan complete.        
    To https://huggingface.co/RJZauner/outputs
       2215eae..03c9511  main -> main
    
    WARNING:huggingface_hub.repository:remote: Scanning LFS files for validity, may be slow...        
    remote: LFS file scan complete.        
    To https://huggingface.co/RJZauner/outputs
       2215eae..03c9511  main -> main
    
    To https://huggingface.co/RJZauner/outputs
       03c9511..31cf7fa  main -> main
    
    WARNING:huggingface_hub.repository:To https://huggingface.co/RJZauner/outputs
       03c9511..31cf7fa  main -> main
    





    'https://huggingface.co/RJZauner/outputs/commit/03c9511b9b420b41259940f0746fe3d023770b9e'


