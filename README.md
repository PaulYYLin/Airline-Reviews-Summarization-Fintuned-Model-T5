# Airline Reviews Summarization Fine-tuned T5 Model

This project fine-tunes a T5 transformer model as a cost-effective alternative to large language models (LLMs) for generating accurate summaries of airline customer reviews. By training on just 2,000 strategically selected reviews, we've created a lightweight model capable of summarizing more than 24,000 reviews with similar quality to advanced LLMs but at a fraction of the computational cost.

The fine-tuned T5 model offers several key advantages:
- **Cost Efficiency**: Dramatically lower inference costs compared to commercial LLMs
- **Speed**: Faster processing time for batch summarization tasks
- **Independence**: No reliance on external API services
- **Consistency**: Predictable outputs with controlled generation parameters

https://drive.google.com/file/d/1iQ3AA4HA6-bjuPLIXaAbiiluwa6ytQ2k/view?usp=sharing

## Project Overview

This project involves a pipeline with several key steps:
1. Data selection and clustering to create a diverse training dataset of airline reviews
2. **LLM-based soft-labeling** to create high-quality summaries for the training data
3. Fine-tuning a T5-small model on the labeled data
4. Evaluating the model's performance against LLM-generated summaries

## Files

The repository contains two main Jupyter notebooks:

### 1. `Summary_Labeling.ipynb`

This notebook handles the preparation of training data:
- Keyword extraction using KeyBERT with the all-MiniLM-L6-v2 model
- Topic clustering using K-means (10 clusters)
- Selecting top 200 reviews from each cluster (2,000 total reviews)
- Creating high-quality summary labels using Grok-2-latest LLM
- Saving the labeled dataset for fine-tuning

### 2. `Model_Finetuning.ipynb`

This notebook handles the model training and evaluation:
- Loading the labeled dataset prepared in the previous step
- Preprocessing the data for T5 model fine-tuning
- Training the T5-small model for summary generation
- Evaluating the model's performance with example reviews
- Comparing the T5-generated summaries with LLM summaries

## Example Results

The fine-tuned T5 model demonstrates its ability to generate concise, informative summaries of airline reviews. For example:

**Original Review:** 
"Worst experience ever, the long delays, to rude staff. The call centre not taking any calls after 5:00pm, although I was informed that they are open 24 hours. Giving vague justification, which was quiet hard to believe, and not letting us know why exactly the flight was cancelled, if this keeps happening the remaining passengers choosing will stop travelling by Srilankan Airlines. Food was good enough, but the staff members at Melbourne airport were on top of their attitude and arrogant."

**LLM Summary:** 
"Srilankan Airlines provided a terrible experience with long delays, rude staff, and a non-responsive call center, though the food was satisfactory."

**T5 Summary:** 
"Srilankan Airlines cancelled a flight after 5:00pm, with rude staff, good food, and arrogant staff at Melbourne airport."

## Technology Used

- **Text Processing**: KeyBERT, scikit-learn
- **LLM for Labeling**: Grok-2-latest
- **Model Training**: HuggingFace Transformers, T5-small
- **Programming**: Python, Pandas, NumPy

## Applications

The fine-tuned model can be used for various applications:
- Summarizing airline customer feedback at scale
- Creating digestible summaries for management reports
- Extracting key insights from large volumes of customer reviews
- Building customer service automation systems
