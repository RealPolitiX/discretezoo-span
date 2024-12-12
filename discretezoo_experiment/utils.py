import torch
import pandas as pd
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset
import datasets
from tqdm import tqdm
from ttp import ttp
import numpy as np

def options(row):
    row_names = ["A", "B", "C", "D"]
    return {rn: row[rn] for rn in row_names}

def yield_answer(row):    
    return row["options"][row["answer_idx"]]

class Formatter:
    def __init__(self, data_frame, column_dict=None, answer_dict=None):
        self.df = data_frame
        if column_dict is None:
            column_dict = {"sent1":"reference",
                           "sent2":"question",
                           "ending0":"A",
                           "ending1":"B",
                           "ending2":"C",
                           "ending3":"D"}
            
        if answer_dict is None:
            answer_dict = {0:"A", 1:"B", 2:"C", 3:"D"}
        
        self.df = self.df.rename(columns=column_dict);
        self.df["answer_idx"] = self.df["label"].map(answer_dict)
        self.df["options"] = self.df.apply(options, axis=1)
        self.df["answer"] = self.df.apply(yield_answer, axis=1)
        self.df["row_num"] = np.arange(len(self.df))
    def export(self, faddress=None, subset=None, orient="index", format="json", **kwargs):
        if subset is not None:
            export_df = self.df.iloc[:subset, :]
        else:
            export_df = self.df
        if format == "json":
            export_df.to_json(faddress, orient=orient, **kwargs)
        elif format == "dict":
            export_df.to_dict(orient=orient, **kwargs)

#preprocess
prefix = "Answer the question without explanation.\n[Context]: "
suffix = "[Answer]: "

def format_question(sample, prefix = prefix, suffix = suffix, incl_qtag=True):
    q = sample["question"]
    spa = "\n"
    formatted_answers = "\n".join([f"{k}: {v}" for k, v in sample["options"].items()])
    if incl_qtag:
        qsent = q.split('. ')[-1]
        q_tagged = q[:-len(qsent)] + "\n[Question]: " + qsent
        q_full = f"""{prefix}{q_tagged}{spa}{formatted_answers} {suffix}"""
    else:
        q_full = f"""{prefix}{q}{spa}{formatted_answers} {suffix}"""

    return q_full

def create_dataset(type, qa_dataset = "usmle"):

    if qa_dataset == "usmle":
        qa = pd.read_parquet(r'./external/medqa_usmle_train_typed.parquet')
    elif qa_dataset == "medmcqa":
        qa = pd.read_parquet(r'./external/medmcqa_train_leaner_typed.parquet')
    else:
        raise ValueError
    
    df = Formatter(qa).df
    if type == 'drugs':
        type = "Drugs?"
    else:
        type = "Diseases?"
    df = df[df[type] == True]
    
    # Convert data structure from DataFrame to Dataset
    ds = datasets.Dataset.from_pandas(df)
    questions = []
    labels = []
    
    for row in tqdm(ds):
        questions.append(format_question(sample = row))
        labels.append(row["label"])
    
    df["formatted_input"] = questions
    print("DF", df["formatted_input"][0])
    print("DS", questions[0])
    
    return Dataset(list(zip(questions, labels))), df

def textmatch(text, template=None):
    
    if template is None:
        if "[Question]" in text:
            template =  """
[Context]: {{ context | _line_ }}
[Question]: {{ q | _line_ }}
A: {{ A | _line_ }}
B: {{ B | _line_ }}
C: {{ C | _line_ }}
D: {{ D | _line_ }} [Answer]:
"""
        else:
            template =  """
[Context]: {{ context | _line_ }}
[Question]: {{ q | _line_ }}
A: {{ A | _line_ }}
B: {{ B | _line_ }}
C: {{ C | _line_ }}
D: {{ D | _line_ }} [Answer]:
"""
    
    parser = ttp(data=text, template=template)
    parser.parse()
    res = parser.result()[0][0]
    
    return res


class Reformatter:

    def __init__(self, text):
        
        letters = ["A", "B", "C", "D"]
        textdict = textmatch(text)
        self.question = textdict['context'] + textdict['q']
        self.options = dict([(x, textdict[x]) for x in letters])

    def reformat(self, prefix=prefix, suffix=suffix, view=None):
        
        qdict = {"question": self.question, "options": self.options}
        text_formatted = format_question(qdict, prefix=prefix, suffix=suffix)
        
        if view == "print":
            print(text_formatted)
        
        return text_formatted

class MedQAWrapper(HuggingFaceModelWrapper):

    def __init__(self, model, tokenizer, key = "text"):
        super().__init__(model, tokenizer)
        self.key = key    
        
    def one_hot_vector(self, out):
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}   
        if len(out) > 1:
            vec = 4
        elif out in label_map:
            vec = label_map[out]
        else:
            vec = 4
        return torch.nn.functional.one_hot(torch.Tensor([vec]).long(), num_classes=5)

    def __call__(self, inputs, model_kwargs = {"max_new_tokens": 256, "do_sample":False},
                 tokenizer_kwargs = {"return_tensors": "pt", "max_length":1024}, decode_outputs = True):
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(inputs, dict):
            inputs = [inputs[self.key]]
        outs = []
        for inpt in inputs:
            if isinstance(inpt, dict):
                inpt = inpt[self.key]
            input_ids = self.tokenizer.encode(inpt, **tokenizer_kwargs).to(self.model.device, non_blocking = True)
            outputs = self.model.generate(input_ids, **model_kwargs)
            
            if decode_outputs:
                outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                out_logits = self.one_hot_vector(outputs) + 1e-8
                outs.append(out_logits.flatten().tolist())
        outs = torch.Tensor(outs)
        return outs

def save_correct_samples(model, ds, filename):
    correct_samples = []
    correct_labels = []
    for text, label in tqdm(ds):
        pred = model(text).argmax().item()
        if pred == label:
            correct_samples.append(text["text"])
            correct_labels.append(label)
    df = pd.DataFrame()
    df["Questions"] = correct_samples
    df["Labels"] = correct_labels
    df.to_csv("usmle_t5_correct.csv")

    ds_correct = datasets.Dataset.from_pandas(df)
    ds_correct.save_to_disk(filename)