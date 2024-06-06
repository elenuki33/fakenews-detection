import spacy
import pandas as pd
from transformers import BartForSequenceClassification, BartTokenizer, AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import pipeline
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from heapq import nlargest
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load('en_core_web_trf')
nlp.add_pipe("merge_entities")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

models = {
        'berta': 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
        'facebook' : 'facebook/bart-large-mnli',
        'distilbert': 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
        'berta-tags' : 'cross-encoder/nli-roberta-base'
        }

# modelos summmarize
# bart 1
tokenizer_bart_summarize  = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model_bart_summarize = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# bart 2
model_bart_summarize2 = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer_bart_summarize2 = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")



DATA = "/home/data-hoaxes.csv"
DATAPATHSAVE = "/home/data-hoaxes-nli.csv"

# Read datasets
df_hoax = pd.read_csv(DATA)
df_hoax.info()

'''
In the dataset we had one fake column (Hoax) and one real column (Fact)
We applied NLI techniques to see if the news were fake or not.
* Fake --> Hoax contradicts Fact
* True --> Hoax no contradcts Facts
'''

df_total = pd.DataFrame(columns=['Hoax', 'Conclusion', 'Fact',  'FactR','label',
                                 'HC_val_facebook', 'HC_tag_facebook', 'HC_val_berta1', 
                                 'HC_tag_berta1', 'HC_val_berta2', 'HC_tag_berta2',
                                 'CF_val_facebook', 'CF_tag_facebook', 'CF_val_berta1', 
                                 'CF_tag_berta1', 'CF_val_berta2', 'CF_tag_berta2',                      
                                 'HF_val_facebook', 'HF_tag_facebook', 'HF_val_berta1', 
                                 'HF_tag_berta1', 'HF_val_berta2', 'HF_tag_berta2',
                                 'HFR_val_facebook', 'HFR_tag_facebook', 'HFR_val_berta1', 
                                 'HFR_tag_berta1', 'HFR_val_berta2', 'HFR_tag_berta2',
                                 'HFR2_val_facebook', 'HFR2_tag_facebook', 'HFR2_val_berta1', 
                                 'HFR2_tag_berta1', 'HFR2_val_berta2', 'HFR2_tag_berta2',
                                 'HFR3_val_facebook', 'HFR3_tag_facebook', 'HFR3_val_berta1', 
                                 'HFR3_tag_berta1', 'HFR3_val_berta2', 'HFR3_tag_berta2'])


def calculate_berta_nli(nli_model, hoax, fact):
    tokenizer = AutoTokenizer.from_pretrained(models[nli_model])
    model = AutoModelForSequenceClassification.from_pretrained(models[nli_model])

    input_ids = tokenizer(fact, hoax, truncation=True, return_tensors="pt")
    output = model(input_ids["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
   
    m = max(prediction)
    m = prediction.index(m)
       
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: round(float(pred), 1) for pred, name in zip(prediction, label_names)}
   
    
    return prediction[label_names[m]] , label_names[m]
    

def calculate_face_nli(nli_model, hoax, fact):
    tokenizer = AutoTokenizer.from_pretrained(models[nli_model])
    model = AutoModelForSequenceClassification.from_pretrained(models[nli_model])
    
    # run through model pre-trained on MNLI
    x = tokenizer.encode(hoax, fact, return_tensors='pt', truncation=True)
    logits = model(x.to(device))[0]
    
    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true 
    entail_contradiction_logits = logits[:,[0,2]]
    #print('entail_contradiction_logits', entail_contradiction_logits)
    probs = logits.softmax(dim=1).tolist()
    
    m = max(probs[0])
    m_i = probs[0].index(m)
    label_names = ["contradiction",  "neutral", "entailment" ]
   
    return probs[0][m_i], label_names[m_i]

        
def calculate_tag_nli(nli_model, hoax, fact):
    tokenizer = AutoTokenizer.from_pretrained(models[nli_model])
    model = AutoModelForSequenceClassification.from_pretrained(models[nli_model])

    features = tokenizer(hoax, fact,  padding=True, truncation=True, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        probs = scores.softmax(dim=1).tolist()
        
        m = max(probs[0])
        m_i = probs[0].index(m)
        
        label_names = ['contradiction', 'entailment', 'neutral']
        
        return   probs[0][m_i], label_names[m_i]

def summarize(text, n=3): # n = num máximo de líneas del resumen
    sent_tokens = sent_tokenize(text)
    word_tokens = word_tokenize(text.lower())
    
    stop_words = set(stopwords.words('spanish'))
    word_tokens = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    
    
    word_freq = FreqDist(word_tokens)
    
    sent_scores = {}
    for sent in sent_tokens:
        for word in word_tokenize(sent.lower()):
            if word in word_freq:
                if len(sent.split()) < 30:
                    if sent not in sent_scores:
                        sent_scores[sent] = word_freq[word]
                    else:
                        sent_scores[sent] += word_freq[word]
    
    summary_sents = nlargest(n, sent_scores, key=sent_scores.get)
    
    
    summary = ' '.join(summary_sents)
    return summary


def summarize_bart(text):
    inputs = tokenizer_bart_summarize(text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")

    summary_ids = model_bart_summarize.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)

    summary = tokenizer_bart_summarize.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary
    

def summarize_bart2(text):
    inputs = tokenizer_bart_summarize2(text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")

    summary_ids = model_bart_summarize2.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)

    summary = tokenizer_bart_summarize2.decode(summary_ids[0], skip_special_tokens=True)

    return summary


for index, row in df_hoax.iterrows(): 
    print('*', index)
    hoax = row['Hoax']
    conclusion = row['Conclusion']
    fact = row['Fact']
    label = row['label']
    
    # Fact resumen
    fact_resumen = summarize(fact) # nltk
    fact_resumen2 = summarize_bart(fact) # bart 1
    fact_resumen3 = summarize_bart2(fact) # bart 2
    
    # HC
    HC_facebook = calculate_face_nli('facebook', hoax, conclusion)
    HC_berta1 = calculate_berta_nli('berta', hoax, conclusion)
    HC_berta2 = calculate_tag_nli('berta-tags', hoax, conclusion)
    
    # CF
    CF_facebook = calculate_face_nli('facebook', conclusion, fact)
    CF_berta1 = calculate_berta_nli('berta', conclusion, fact)
    CF_berta2 = calculate_tag_nli('berta-tags', conclusion, fact)
    
    # HF
    HF_facebook = calculate_face_nli('facebook', hoax, fact)
    HF_berta1 = calculate_berta_nli('berta', hoax, fact)
    HF_berta2 = calculate_tag_nli('berta-tags', hoax, fact)
    
    '''
    # HF resumen nltk
    HFR_facebook = calculate_face_nli('facebook', hoax, fact_resumen)
    HFR_berta1 = calculate_berta_nli('berta', hoax, fact_resumen)
    HFR_berta2 = calculate_tag_nli('berta-tags', hoax, fact_resumen)
    
    # HF resumen facebook bart 1
    HFR2_facebook = calculate_face_nli('facebook', hoax, fact_resumen2)
    HFR2_berta1 = calculate_berta_nli('berta', hoax, fact_resumen2)
    HFR2_berta2 = calculate_tag_nli('berta-tags', hoax, fact_resumen2)
    
    # HF resumen facebook bart 2
    HFR3_facebook = calculate_face_nli('facebook', hoax, fact_resumen3)
    HFR3_berta1 = calculate_berta_nli('berta', hoax, fact_resumen3)
    HFR3_berta2 = calculate_tag_nli('berta-tags', hoax, fact_resumen3)
    '''
    
    # Define the new row to be added
    new_row = {
            'Hoax': hoax, 
            'Conclusion': conclusion, 
            'Fact': fact, 
            'FactR': fact_resumen,
            'label':label,
                                 
            'HC_val_facebook': HC_facebook[0], 
            'HC_tag_facebook': HC_facebook[1],                                  
            'HC_val_berta1': HC_berta1[0], 
            'HC_tag_berta1': HC_berta1[1], 
            'HC_val_berta2': HC_berta2[0], 
            'HC_tag_berta2': HC_berta2[1],
                                 
            'CF_val_facebook': CF_facebook[0], 
            'CF_tag_facebook': CF_facebook[1], 
            'CF_val_berta1': CF_berta1[0], 
            'CF_tag_berta1': CF_berta1[1], 
            'CF_val_berta2': CF_berta2[0], 
            'CF_tag_berta2': CF_berta2[1],
                                 
            'HF_val_facebook': HF_facebook[0], 
            'HF_tag_facebook': HF_facebook[1], 
            'HF_val_berta1': HF_berta1[0], 
            'HF_tag_berta1': HF_berta1[1], 
            'HF_val_berta2': HF_berta2[0], 
            'HF_tag_berta2': HF_berta2[1],
            
            
            }
     '''
     'HFR_val_facebook': HFR_facebook[0], 
            'HFR_tag_facebook': HFR_facebook[1], 
            'HFR_val_berta1': HFR_berta1[0], 
            'HFR_tag_berta1': HFR_berta1[1], 
            'HFR_val_berta2': HFR_berta2[0], 
            'HFR_tag_berta2': HFR_berta2[1],
            
            'HFR2_val_facebook': HFR2_facebook[0], 
            'HFR2_tag_facebook': HFR2_facebook[1], 
            'HFR2_val_berta1': HFR2_berta1[0], 
            'HFR2_tag_berta1': HFR2_berta1[1], 
            'HFR2_val_berta2': HFR2_berta2[0], 
            'HFR2_tag_berta2': HFR2_berta2[1],
                        
            'HFR3_val_facebook': HFR3_facebook[0], 
            'HFR3_tag_facebook': HFR3_facebook[1], 
            'HFR3_val_berta1': HFR3_berta1[0], 
            'HFR3_tag_berta1': HFR3_berta1[1], 
            'HFR3_val_berta2': HFR3_berta2[0], 
            'HFR3_tag_berta2': HFR3_berta2[1]
    '''
    # Add the new row to the DataFrame
    df_total.loc[len(df_total)] = new_row
    

df_total.info()
df_total.to_csv(DATAPATHSAVE)

