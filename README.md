# RDQuantization_v1
Quantiative Assessment of Respiratory Distress Severity

1.	Definition: Respiratory Distress is usually characterized by laboured breathing or signs that patient is not getting enough oxygen. Assessment of respiratory distress (RD) is crucial not only from the perspective of determining clinical stability of patient but is also linked with the initiation of resuscitation efforts by rapid response team in hospitals. Rising severity of RD is often considered as harbinger of other life-threatening conditions like acute respiratory distress syndrome (ARDS), cardiac arrest, respiratory failure etc

2.	Approach Developed: Quantitative Assessment of Respiratory Distress. It gives a score from 0 to 1 based on severity of RD. Can work with streaming monitor arrangement.

3.	Parameters Used: RR, SpO2           OR              RR, SpO2, HR 

4.	Target Patients: Patients not on ventilator

5.	Sampling Rate: 1 sample/minute

6.	Database Used for Present Model: MIMIC-III

7.	Annotations: Clinical annotations of about 200 records

8.	Algorithms: Logistic Regression, Support Vector Machine, Multilayer Perceptron, Convolutional Neural Network, Symbolic Aggregate Approximation 

9.	Possible Deployment: Platform on which streaming data of these 2-3 parameters is available. 

a.	AMS (Ambulatory Monitoring System, RR & SpO2 are available)

10.	 We can incorporate OIRD (Opioid Induced Respiratory Depression)

