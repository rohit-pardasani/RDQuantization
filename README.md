<h1> Quantiative Assessment of Respiratory Distress Severity</h1> <br> <b>Machine Learning and Deep Learning Approachesto Quantify Respiratory Distress Severity andPredict Critical Alarms</b>

1.	Definition: Respiratory Distress is usually characterized by laboured breathing or signs that patient is not getting enough oxygen. Assessment of respiratory distress (RD) is crucial not only from the perspective of determining clinical stability of patient but is also linked with the initiation of resuscitation efforts by rapid response team in hospitals. Rising severity of RD is often considered as harbinger of other life-threatening conditions like acute respiratory distress syndrome (ARDS), cardiac arrest, respiratory failure etc

2.	Approach Developed: Quantitative Assessment of Respiratory Distress. It gives a score from 0 to 1 based on severity of RD. Can work with streaming monitor arrangement.

3.	Parameters Used: RR, SpO2

4.	Target Patients: Patients not on ventilator

5.	Sampling Rate: 1 sample/minute

6.	Database Used for Present Model: MIMIC-III

7.	Annotations: Clinical annotations of 912 records

8.	Algorithms: Logistic Regression, Support Vector Machine, Multilayer Perceptron, Convolutional Neural Network (CNN), Decision Trees, Long Short Term Memory (LSTM)

9.	Possible Deployment: Platform on which streaming data of these 2 parameters is available. 
    a.	Ambulatory Monitoring System where RR & SpO2 are available
    b.  ICU Monitoring System
