{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e602ab67-e6de-45ef-87e1-c497183c106c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#loading the required libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import streamlit as st\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.datasets import imdb\n",
    "from tensorflow.keras.preprocessing import sequence\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import SimpleRNN,Embedding,Dense\n",
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "from tensorflow.keras.models import load_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "365fe23f-5142-48eb-8e78-7bb986526661",
   "metadata": {},
   "outputs": [],
   "source": [
    "word_index=imdb.get_word_index()\n",
    "reverse_word_index={value:key for key,value in word_index.items()}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c548dca7-be98-4cd3-88a3-9eb0fe9f38e4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    }
   ],
   "source": [
    "#loading the model\n",
    "model=load_model('RNN_Sentiment.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c81811be-08be-4279-abf9-5457f056ebbd",
   "metadata": {},
   "outputs": [],
   "source": [
    "#function to decode the review\n",
    "def decode_review(encoded_review):\n",
    "    return (' '.join([reverse_word_index.get(i-3,'?') for i in sample_review]))\n",
    "\n",
    "#function to pre process the user review\n",
    "def preprocess_review(text):\n",
    "    words=text.lower().split()\n",
    "    encoded_review=[word_index.get(word,2)+3 for word in words]\n",
    "    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)\n",
    "    return(padded_review)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "c9a19caf-b05f-409a-af2b-ff06e01f5c2f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#function to create the prediction fun\n",
    "def predict_sentiment(review):\n",
    "    preprocessed_input=preprocess_review(review)\n",
    "    prediction=model.predict(preprocessed_input)\n",
    "    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'\n",
    "    return (sentiment,prediction[0][0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "61e653ed-e964-4aa7-bb65-1ecbbbabe464",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Design Streamlit app\n",
    "st.title('IMDB Movie Review Sentiment Analysis')\n",
    "st.write('Input a Movie review to classify as positive or negative sentiment')\n",
    "user_input=st.text_area('Movie Review')\n",
    "if st.button('Classify'):\n",
    "    preprocess_input=preprocess_review(user_input)\n",
    "    prediction=model.predict(preprocess_input)\n",
    "    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'\n",
    "    st.write(f'sentiment :{sentiment}')\n",
    "    st.write(f'score:{prediction[0][0]}')\n",
    "else:\n",
    "    st.write('Please enter a movie review')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
