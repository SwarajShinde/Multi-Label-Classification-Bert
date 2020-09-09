class BertDataset(Dataset):
    def __init__(self,reviews,targets):
        self.reviews=reviews
        self.tokenizer = tokenizer
        self.targets = targets # have a doubt over this 
        self.max_len = MAX_LEN
        
    def __len__(self):
        return len(self.reviews)


    def __getitem__(self,item):
        review = str(self.reviews[item])
      #  review = "".join(review.split())
        inputs = self.tokenizer.encode_plus( 
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation = True
        )
        target = self.targets[item]


        ids =  inputs['input_ids']
        mask = inputs['attention_mask']



        return {
            'review_text':review,
            'ids':inputs['input_ids'].flatten(),
            'mask':inputs['attention_mask'].flatten(),
            'target':torch.tensor(target,dtype=torch.long)
        }
        
from sklearn.model_selection import train_test_split
df_train,df_valid = train_test_split(train,test_size=0.1,random_state=42,stratify = train['Sentiment'])
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)


train_dataset = BertDataset(reviews=df_train.review.values,targets=df_train.Sentiment.values)
valid_dataset = BertDataset(reviews=df_valid.review.values,targets=df_valid.Sentiment.values)

train_dataloader = DataLoader(train_dataset,batch_size=train_batch_size,num_workers=4)
valid_dataloader = DataLoader(valid_dataset,batch_size=valid_batch_size,num_workers=1)


# Test DataLoaders Now


class BertDataset_test(Dataset):
    def __init__(self,reviews):
        self.reviews=reviews
        self.tokenizer = tokenizer
        self.max_len = MAX_LEN
        
    def __len__(self):
        return len(self.reviews)


    def __getitem__(self,item):
        review = str(self.reviews[item])
        inputs = self.tokenizer.encode_plus( 
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation = True
        )


        ids =  inputs['input_ids']
        mask = inputs['attention_mask']



        return {
            'review_text':review,
            'ids':inputs['input_ids'].flatten(),
            'mask':inputs['attention_mask'].flatten()
        }
        
test_df = BertDataset_test(test.review.values)

test_dataloader = DataLoader(test_df,batch_size=4,num_workers=2)

'''
example = next(iter(test_dataloader))
example.keys()
'''

