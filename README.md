# Company Bankruptcy Prediction!
This project is a collaboration between the Pacmann Data School's Introduction to Machine Learning (IML) and Machine Learning Process (MLP) courses.

## General Introductions
Financial risk prediction is an important topic in financial analysis because it can assist businesses in reducing financial distress and taking appropriate actions in the future. Bankruptcy prediction has been a topic of discussion for nearly a century, and it remains one of the hottest topics in economics. The goal of predicting financial institutions' distress is to create a predictive model that combines various econometric measures and allows one to forecast a firm's financial situation at least 1 year in the future. Predicting corporate bankruptcy is critical for both investors and businesses.

##  Introduction to Machine Learning (IML) 
The IML project is primarily concerned with machine learning modeling experiments. 
The **2 notebooks** folder contains the source code for the modeling and testing scenarios that were completed.
See: for a detailed modeling 
> https://docs.google.com/document/d/1wBAJakg23FQ721Ep7cdm6Ab7pohvkP0rBosVS8d8siY/edit?usp=sharing

##  Machine Learning Process (MLP).
The MLP project focuses on end-to-end machine learning workflows, beginning with understanding business problems, deriving business metrics, machine learning metrics, and creating fire services for the backend and front end services until they are ready for server deployment. Details about the report can be found at:
> https://docs.google.com/document/d/10AAaHIiPcAMV1gkTiMz54yJ8PeRf4Cv_hlIvaSW_DGQ/edit?usp=sharing

### How to run API and Streamlit (Front End) Services
> 1. Check that you have **Docker** installed. If not, search for "How to install Docker on Windows/Ubuntu/Os that you used."

> 2. Make sure you have **Git** installed as well. If not, search for "How to install git on Windows/Ubuntu/Os that you used."

> 3. Make a clone of this repository : 
`git clone  https://github.com/bimapriambodo/Company_Bankruptcy_Prediction-.git`

> 4. Navigate to the cloned folder's directory. Open a CMD terminal and compile the docker file as follows:
`docker-compose up`
Wait until everything is completed. If your internet connection is fast, the process will be as well.

>5. To test Api, open a browser and enter the following address :
`http://0.0.0.0:8080/docs`

>6. To try Streamlit, launch a browser and type the following address :
`http://172.18.0.3:8501/`

Congratulations, you are now free to use the application !!!


### Input API Data

Input Api uses json format like this:
```JSON
{
  "Attr21": 0.0,
  "Attr13": 0.0,
  "Attr27": 0.0,
  "Attr34": 0.0,
  "Attr24": 0.0,
  "Attr35": 0.0,
  "Attr51": 0.0,
  "Attr6": 0.0,
  "Attr56": 0.0,
  "Attr49": 0.0,
  "Attr40": 0.0,
  "Attr55": 0.0,
  "Attr50": 0.0,
  "Attr58": 0.0,
  "Attr43": 0.0
}
  ```
___

| VARIABLES        | DATA TYPE           |
| ------------- |:-------------:| 
| Attr21      | FLOAT | 
| Attr13      | FLOAT      |
| Attr27 | FLOAT      | 
| Attr34 | FLOAT |
| Attr24 | FLOAT |
| Attr35 | FLOAT |
| Attr51 | FLOAT |
| Attr6 | FLOAT |
| Attr56 | FLOAT |
| Attr49 | FLOAT |
| Attr40 | FLOAT |
| Attr55 | FLOAT |
| Attr50 | FLOAT |
| Attr58 | FLOAT |
| Attr43 | FLOAT |
___

## Future Works

Many scenarios have not yet been tried and would be quite interesting iterative combinations to investigate as material for future works, for example,

1.  We do not remove outliers from the data in this experiment because we believe outliers are a natural phenomenon and provide information. However, what can be tested in future works if the outliers are removed?
    
2.  Standard scalers are not the only option for standardizing data. There is a min/max scaler as well as other methods. What do you think if you compare it with other methods? Is there any other influence? It's possible.
    
3.  Not only are machine learning methods used in this trial, but many are widely used, such as neural networks, Cat-boost, light GBM, and others. Can try to be compared, it appears to be interesting.
    
4.  We do not perform feature engineering or variable reduction based on existing variables. Then, for example, try dimensional reduction using PCA, how does it work?

5. Deploy to Cloud Services ! 

We hope that this trial will provide an overview of highly influential iterative processes in data science. Matt, stay hungry and foolish!

##  Rerence
Chang, Hanxu. 2019. The Application of Machine Learning Models in Company Bankruptcy Prediction. ICSEB: International Conference on Software and e-Business. Tokyo: Japan

Zieba,Maciej. 2016. Ensemble Boosted Trees with Synthetic Features Generation in Application to Bankruptcy Prediction. Expert Systems with Applications: An International Journal

Thi Kha Nguyen. 2018. PREDICTING BANKRUPTCY USING MACHINE LEARNING ALGORITHMS. JOURNAL OF SCIENCE AND TECHNOLOGY, NO. 12(133). THE UNIVERSITY OF DANANG

[https://chirag-sehra.medium.com/decision-trees-explained-easily-28f23241248](https://chirag-sehra.medium.com/decision-trees-explained-easily-28f23241248)

[https://towardsdatascience.com/understanding-random-forest-58381e0602d2#:~:text=The%20random%20forest%20is%20a,that%20of%20any%20individual%20tree](https://towardsdatascience.com/understanding-random-forest-58381e0602d2#:~:text=The%20random%20forest%20is%20a,that%20of%20any%20individual%20tree).

[https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-alalgorithm-6a6e71d01761](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)

[https://www.ibm.com/id-en/topics/knn#:~:text=The%20k%2Dnearest%20neighbors%20algorithm%2C%20also%20known%20as%20KNN%20or,of%20an%20individual%20data%20point](https://www.ibm.com/id-en/topics/knn#:~:text=The%20k%2Dnearest%20neighbors%20algorithm%2C%20also%20known%20as%20KNN%20or,of%20an%20individual%20data%20point).

[https://ai.plainenglish.io/xgboost-in-a-nutshell-211e170e8b48](https://ai.plainenglish.io/xgboost-in-a-nutshell-211e170e8b48)

[https://towardsdatascience.com/curse-of-dimensionality-a-curse-to-machine-learning-c122ee33bfeb](https://towardsdatascience.com/curse-of-dimensionality-a-curse-to-machine-learning-c122ee33bfeb)
