import streamlit as st
from joblib import load
import pandas as pd
import numpy as np

from config import BASIC_MODEL_PATH
from utils import user_predict

st.title("Determine someone makes over $50K a year")

with st.form("my_form"):
	data = dict()

	st.write("Please provide input data to get prediction")
	data["age"] = st.number_input(label="Age", min_value=1, max_value=120)
	data["workclass"] = st.selectbox(label="Workclass", 
		options=["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
	data["fnlwgt"] = st.number_input(label="Final weight", min_value=1, max_value=1000000)
	data["education"] = st.selectbox(label="Education",
		options=['HS-grad','Some-college','1st-4th','5th-6th','7th-8th','9th','10th','11th',
				'12th','Doctorate','Prof-school',
 				'Bachelors','Masters','Assoc-acdm','Assoc-voc','Preschool'])
	data["education.num"] = st.number_input(label="Education num", min_value=0, max_value=50)
	data["marital.status"] = st.selectbox(label="Marital status",
		options=['Widowed','Divorced','Separated','Never-married','Married-civ-spouse',
				'Married-spouse-absent','Married-AF-spouse'])
	data["occupation"] = st.selectbox(label="Occupation",
		options=['Exec-managerial','Machine-op-inspct','Prof-specialty',
 				'Other-service','Adm-clerical','Craft-repair','Transport-moving',
 				'Handlers-cleaners','Sales','Farming-fishing','Tech-support',
 				'Protective-serv','Armed-Forces','Priv-house-serv'])
	data["relationship"] = st.selectbox(label="Relationship",
		options=['Not-in-family','Unmarried','Own-child','Husband','Wife','Other-relative'])
	data["race"] = st.selectbox(label="Race",
		options=['White','Black','Asian-Pac-Islander','Amer-Indian-Eskimo', 'Other'])
	data["sex"] = st.selectbox(label="Sex",
		options=['Female','Male'])
	data["capital.gain"] = st.number_input(label="Capital gain", min_value=0, max_value=10000)
	data["capital.loss"] = st.number_input(label="Capital loss", min_value=0, max_value=10000)
	data["hours.per.week"] = st.number_input(label="Number work per week", min_value=1, max_value=150)
	data["native.country"] = st.selectbox(label="Native country",
		options=['United-States','Mexico','Greece','Vietnam','China','Taiwan','India',
 				'Philippines','Trinadad&Tobago','Canada','South','Holand-Netherlands',
 				'Puerto-Rico','Poland','Iran','England','Germany','Italy','Japan','Hong',
 				'Honduras','Cuba','Ireland','Cambodia','Peru','Nicaragua',
 				'Dominican-Republic','Haiti','El-Salvador','Hungary','Columbia',
 				'Guatemala','Jamaica','Ecuador','France','Yugoslavia','Scotland',
 				'Portugal','Laos','Thailand','Outlying-US(Guam-USVI-etc)'])
	submitted = st.form_submit_button("Submit")
	if submitted:
		# print(data)
		input_data = pd.DataFrame().append(data, ignore_index = True)
		result = user_predict(BASIC_MODEL_PATH, input_data)
		st.title("Result: {}".format(result))