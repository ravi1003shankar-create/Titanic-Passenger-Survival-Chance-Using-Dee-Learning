import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle


st.title("Passenger Survival Chance in the Titanic Journey")

pclass=st.slider("Enter the Passenger Class",1,3)
sex=st.selectbox("Enter the Passenger Gender",['male','female'])
SibSp=st.slider("Enter the total no. of Passenger Sibling and Spouse",1,8)
Parch=st.slider("Enter the total no. of Parents and Child",0,6)
Fare=st.number_input("Enter the Fare of the Passenger")
embarked=st.selectbox("Enter the Station where the passenger had started the Journey",['Southampton','Chebourg','Queenstown'])

data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':SibSp,'Parch':Parch,'Fare':Fare, 'Embarked':embarked}])

model=load_model('model.h5')

with open('lable_encoder.pkl','rb') as file:
    label=pickle.load(file)

with open ('one_hot_encoder.pkl','rb') as file:
    one_hot=pickle.load(file)

with open('standard_scaler.pkl','rb') as file:
    scaler=pickle.load(file)

data['Sex']=label.transform(data['Sex'])
embarked=one_hot.transform(data[['Embarked']])


embarked=pd.DataFrame(embarked,columns=one_hot.get_feature_names_out())

data=pd.concat([data.drop(columns=['Embarked']),embarked ],axis=1)

data[['Pclass','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

y=model.predict(data)

y=y[0][0]


def Chance(y):
    if y>0.5:
        return 'The Passenger will survive the Journey'
    else:
        'The Passenger wont survie the journey'



if st.button('Predict_Survival_Chance'):
    st.write('Probability of the survival chance',y)
    st.write(Chance(y))






