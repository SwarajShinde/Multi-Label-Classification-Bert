class BertModel(nn.Module):
    def __init__(self):
        super(BertModel,self).__init__()
        self.bert = transformers.BertModel.from_pretrained(bert_path)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(768,4)
        
    def forward(self,ids,mask):
        _,o2 = self.bert(ids,attention_mask=mask)
        bo = self.drop(o2)
        output = self.out(bo)
        return output
