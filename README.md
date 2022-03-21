# AI for Medicine Professional Certificate
## Offered by DeepLearning.AI

All work in this repository was part of the courses included in the AI for Medicine Professional Certificate, found here: https://www.coursera.org/specializations/ai-for-medicine

The exercises in this repository were completed by Frances Koback for this course. Some of the code in the python notebooks have been altered to work on an updated local python 3.6 environment.

**Instructors of This Course**: `Pranav Rajpurkar, Harvard University`, `Bora Uyumazturk, Machine Learning Engineer, Viaduct`, `Amirhossein Kiani, Product Manager, Google Health` and `Eddy Shyu, Project Lead, DeepLearning.AI`.

# About this Course
"AI is transforming the practice of medicine. It’s helping doctors diagnose patients more accurately, make predictions about patients’ future health, and recommend better treatments. This Specialization will give you practical experience in applying machine learning to concrete problems in medicine.

Medical treatment may impact patients differently based on their existing health conditions. In this third course, you’ll recommend treatments more suited to individual patients using data from randomized control trials. In the second week, you’ll apply machine learning interpretation methods to explain the decision-making of complex machine learning models. Finally, you’ll use natural language entity extraction and question-answering methods to automate the task of labeling medical datasets.

These courses go beyond the foundations of deep learning to teach you the nuances in applying AI to medical use cases. If you are new to deep learning or want to get a deeper foundation of how neural networks work, we recommend that you take the Deep Learning Specialization.

AI is transforming the practice of medicine. It’s helping doctors diagnose patients more accurately, make predictions about patients’ future health, and recommend better treatments. In this Specialization, you’ll gain practical experience applying machine learning to concrete problems in medicine. You’ll learn how to:

- Diagnose diseases from X-Rays and 3D MRI brain images
- Predict patient survival rates more accurately using tree-based models
- Estimate treatment effects on patients using data from randomized trials
- Automate the task of labeling medical datasets using natural language processing"

## Programming Assignments

### Course 1: [AI for Medical Diagnosis](https://www.coursera.org/learn/ai-for-medical-diagnosis)
  
  - **Week 1**
      - [Chest X-Ray Medical Diagnosis with Deep Learning](https://nbviewer.jupyter.org/github/franceskoback/AI-For-Medicine-Specialization/blob/main/AI_For_Medical_Diagnosis/Week_1/Chest_X-Ray_Medical_Diagnosis_with_Deep_Learning.ipynb)
    
  - **Week 2**
      - [Evaluation of Diagnostic Models](https://nbviewer.jupyter.org/github/franceskoback/AI-For-Medicine-Specialization/blob/main/AI_For_Medical_Diagnosis/Week_2/Evaluation_of_Diagnostic_Models.ipynb)  
  - **Week 3**
      - [Brain Tumor Auto-Segmentation for Magnetic Resonance Imaging (MRI)](https://nbviewer.jupyter.org/github/franceskoback/AI-For-Medicine-Specialization/blob/main/AI_For_Medical_Diagnosis/Week_3/Brain_Tumor_Auto-Segmentation_for_Magnetic_Resonance_Imaging_(MRI).ipynb)
     

### Course 2: [AI for Medical Prognosis](https://www.coursera.org/learn/ai-for-medical-prognosis)
  
  - **Week 1**
      - [Diagnosing Diseases using Linear Risk Models](https://nbviewer.jupyter.org/github/franceskoback/AI-For-Medicine-Specialization/blob/main/AI_For_Medical_Prognosis/Week_1/Build_and_Evaluate_a_Linear_Risk_Model.ipynb)
   
  - **Week 2** 
      - [Risk Models Using Machine Learning](https://nbviewer.jupyter.org/github/franceskoback/AI-For-Medicine-Specialization/blob/main/AI_For_Medical_Prognosis/Week_2/Risk_Models_Using_Tree-based_Models.ipynb)
  
  - **Week 3** 
      - [Non-Parametric Estimators for Survival Analysis](https://nbviewer.jupyter.org/github/https://github.com/franceskoback/AI-For-Medicine-Specialization/blob/main/AI_For_Medical_Prognosis/Week_3/Survival_Estimates_that_Vary_with_Time.ipynb)

  - **Week 4** 
      - [Cox Proportional Hazards and Random Survival Forests](https://nbviewer.jupyter.org/github/franceskoback/AI-For-Medicine-Specialization/blob/main/AI_For_Medical_Prognosis/Week_4/Cox_Proportional_Hazards_and_Random_Survival_Forests.ipynb)
 

### Course 3: [AI For Medical Treatment](https://www.coursera.org/learn/ai-for-medical-treatment)
  - **Week 1** 
      - [Estimating Treatment Effect Using Machine Learning](https://nbviewer.jupyter.org/github/franceskoback/AI-For-Medicine-Specialization/blob/main/AI_For_Medical_Treatment/Week_1/Estimating_Treatment_Effect_Using_Machine_Learning.ipynb)
   
  - **Week 2** 
      - [Natural Language Entity Extraction](https://nbviewer.jupyter.org/github/franceskoback/AI-For-Medicine-Specialization/blob/main/AI_For_Medical_Treatment/Week_3/Model_Interpretation_Methods.ipynb)
 
  - **Week 3** 
      - [ML Interpretation](https://nbviewer.jupyter.org/github/franceskoback/AI-For-Medicine-Specialization/blob/main/AI_For_Medical_Treatment/Week_3/Model_Interpretation_Methods.ipynb)

## Syllabus
Text here is quoted from Coursera's syllabus for this course.

### Course 1: AI For Medical Diagnosis

How can AI be applied to medical imaging to diagnose diseases? In this first course, you’ll learn about the nuances of working with both 2D and 3D medical image data, for multi-class classification and image segmentation. You’ll then apply what you’ve learned to classify diseases in X-Ray images and segment tumors in 3D MRI brain images. Finally, you’ll learn how to properly evaluate the performance of your models.

#### Week 1:
- Introduction: A conversation with Andrew Ng
- Diagnosis examples
- Model training on chest X-Rays
- Training, prediction, and loss
- Class imbalance
- Binary cross entropy loss function
- Resampling methods
- Multi-task loss
- Transfer learning and data augmentation
- Model testing

#### Week 2:
- Introduction: A conversation with Andrew Ng
- Evaluation metrics
- Accuracy in terms of conditional probability
- Sensitivity, specificity, and prevalence
- Confusion matrix
- ROC curve
- Threshold (operating point)
- Confidence intervals
- Width of confidence intervals and sample size
- Using a sample to estimate the population 

#### Week 3:
- Introduction: A conversation with Andrew Ng
- Representing MRI data
- Image registration
- 2D and 3D segmentation
- 3D U-Net
- Data augmentation for segmentation
- Loss function for image segmentation
- Soft dice loss
- External validation
- Retrospective vs. prospective data
- Working with cleaned vs. raw data
- Measuring patient outcomes
- Algorithmic bias
- Model influence on medical decision-making

---

### Course 2: AI For Medical Prognosis

Machine learning is a powerful tool for prognosis, a branch of medicine that specializes in predicting the future health of patients. First, you’ll walk through multiple examples of prognostic tasks. You’ll then use decision trees to model non-linear relationships, which are commonly observed in medical data, and apply them to predicting mortality rates more accurately. Finally, you’ll learn how to handle missing data, a key real-world challenge.

#### Week 1:
- Introduction: A conversation with Andrew Ng
- Examples of prognostic tasks
- Patient profile to risk score
- Risk score for atrial fibrillation
- Liver disease mortality
- Calculate 10-year risk of heart disease
- Risk score computation
- Evaluating prognostic models
- Concordant pairs
- Risk ties
- Permissible pairs
- C-index interpretation

#### Week 2:
- Decision trees for prognosis
- Predicting mortality risk
- Dividing the input space
- Non-linear associations
- Class boundaries of a decision tree
- Random forest
- Ensemble methods
- Survival data
- Problems with dropping incomplete rows
- Dropping incomplete case changes the distribution
- Imputation
- Mean imputation
- Regression imputation

#### Week 3:
- Survival function
- Censoring
- Collecting time data
- Heart attack data
- Estimating the survival function
- Using censored data
- Chain rule of conditional probability
- Derivation
- Calculating probabilities from the data
- Comparing estimates
- Kaplan Meier Estimate
 
#### Week 4:
- Hazard functions
- Survival to hazard
- Cumulative hazard
- Individualized predictions
- Individual vs. baseline hazard
- Smoker vs. non-smoker
- Effect of age on hazard
- Factor risk increase or decrease
- Survival trees
- Nelson Aelen estimator
- Mortality score
- Evaluating survival models
- Permissible pair examples
- Harrell’s concordance index

---

### Course 3: AI For Medical Treatment

Medical treatment may impact patients differently based on their existing health conditions. In this final course, you’ll estimate treatment effects using data from randomized control trials and applying tree-based models. In the second week, you’ll apply machine learning interpretation methods to explain the decision-making of complex machine learning models. In the final week of this course, you’ll use natural language entity extraction and question-answering methods to automate the task of labeling medical datasets.

#### Week 1:
- Treatment effect estimation
- Randomized control trials
- Average risk reductio
- Individualized treatment effect
- T-Learner and S-Learner
- C-for-benefit

#### Week 2:
- Information extraction from medical reports
- Rules-based label extraction
- Text matching
- Negation detection
- Dependency parsing
- Question-Answering with BERT
 
#### Week 3:
- Machine Learning Interpretation
- Interpret CNN models with GradCAM
- Aggregate and Individual feature importance
- Permutation Importance
- Shapley Values
- Interpret random forest models

## Credits

The readme for this page was designed following the layout of [Aman Chada's](https://github.com/amanchadha/coursera-ai-for-medicine-specialization) GitHub Page for this Course.
All work on the code contained in these folders are my own, please do not copy and paste. 
