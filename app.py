import numpy as np
import torch
from pytorch_transformers import BertTokenizer, BertForMaskedLM
import nltk
import streamlit as st
nltk.download('punkt')
from PIL import Image # for importing image


# FOOTER : https://discuss.streamlit.io/t/streamlit-footer/12181
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p> <b>Based on GitHub repository</b>  <a style='display: block; text-align: center;' href="https://github.com/prakhar21/Fill-in-the-BERT.git" target="_blank">Fill-in-the-BERT by prakhar21</a></p>
</div>
"""


# Predicting a word
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=True)
model.eval()
def predict():
	sentence_orig = st.text_input('Input text:', 'I __ you')
	if '__' not in sentence_orig:
		return sentence_orig

	sentence = sentence_orig.replace('__', 'MASK')
	tokens = nltk.word_tokenize(sentence)
	sentences = nltk.sent_tokenize(sentence)
	sentence = " [SEP] ".join(sentences)
	sentence = "[CLS] " + sentence + " [SEP]"
	tokenized_text = tokenizer.tokenize(sentence)
	masked_index = tokenized_text.index('mask')
	tokenized_text[masked_index] = "[MASK]"
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

	segments_ids = []
	sentences = sentence.split('[SEP]')
	for i in range(len(sentences)-1):
		segments_ids.extend([i]*len(sentences[i].strip().split()))
		segments_ids.extend([i])

	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])

	with torch.no_grad():
	    outputs = model(tokens_tensor, token_type_ids=segments_tensors) 
	    predictions = outputs[0] 
	    attention = outputs[-1] 

	dim = attention[2][0].shape[-1]*attention[2][0].shape[-1]
	a = attention[2][0].reshape(12, dim)
	b = a.mean(axis=0)
	c = b.reshape(attention[2][0].shape[-1],attention[2][0].shape[-1])
	avg_wgts = c[masked_index]
	#print (avg_wgts, tokenized_text)
	focus = [tokenized_text[i] for i in avg_wgts.argsort().tolist()[::-1] if tokenized_text[i] not in ['[SEP]', '[CLS]', '[MASK]']][:5]

	predicted_index = torch.argmax(predictions[0, masked_index]).item()
	predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

	for f in focus:
		sentence_orig = sentence_orig.replace(f, '<font color="blue">'+f+'</font>')
	return sentence_orig.replace('__', '<font color="red"><b><i>'+predicted_token+'</i></b></font>')


if __name__=='__main__':
	st.header(":blue[Fill-mask model]")
	st.write(":blue[Fill-mask model] is a fill-in-the-blanks model that is trained to predict the missing word in the sentence. For the purpose of this demo we will be using pre-trained **distillbert-base-uncased** as our prediction model.")

	# Print prediction
	predicted_sen = predict()
	st.markdown(predicted_sen, unsafe_allow_html=True)
	
	# Import image to centre of page : https://stackoverflow.com/questions/70932538/how-to-center-the-title-and-an-image-in-streamlit
	# image = Image.open('BERT_img_test.png')
	col1, col2, col3 = st.columns(3)
	with col1:
		st.write(' ')
	with col2:
		st.image("BERT_img_test.png")
	with col3:
		st.write(' ')

	# Demo munual
	st.write("**Demo Manual**")
	st.write("1. Use __ as the marker for representing blank space in the text.\n2. Give only one blank space (__) at a time. More than one blanks are not taken care.")

	# Insert footer
	st.markdown(footer, unsafe_allow_html=True)
