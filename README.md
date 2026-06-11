# Automated MLOps Pipeline Project

## Business Context

**"Visit with Us"**, a leading travel company, is revolutionizing the tourism industry by leveraging data-driven strategies to optimize operations and customer engagement. While introducing a new package offering, such as the Wellness Tourism Package, the company faces challenges in targeting the right customers efficiently. The manual approach to identifying potential customers is inconsistent, time-consuming, and prone to errors, leading to missed opportunities and suboptimal campaign performance.

To address these issues, the company aims to implement a scalable and automated system that integrates customer data, predicts potential buyers, and enhances decision-making for marketing strategies. By utilizing an MLOps pipeline, the company seeks to achieve seamless integration of data preprocessing, model development, deployment, and CI/CD practices for continuous improvement. This system will ensure efficient targeting of customers, timely updates to the predictive model, and adaptation to evolving customer behaviors, ultimately driving growth and customer satisfaction.

---

## Objective

As an MLOps Engineer at "Visit with Us," your responsibility is to design and deploy an MLOps pipeline on GitHub to automate the end-to-end workflow for predicting customer purchases. The primary objective is to build a model that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them. The pipeline will include data cleaning, preprocessing, transformation, model building, training, evaluation, and deployment, ensuring consistent performance and scalability. By leveraging GitHub Actions for CI/CD integration, the system will enable automated updates, streamline model deployment, and improve operational efficiency. This robust predictive solution will empower policymakers to make data-driven decisions, enhance marketing strategies, and effectively target potential customers, thereby driving customer acquisition and business growth.

---

## Project Scope

The MLOps pipeline automates the following stages:

1. Data ingestion and validation
2. Data cleaning and preprocessing
3. Feature engineering and transformation
4. Model training and evaluation
5. Model deployment readiness
6. CI/CD automation for continuous integration and updates

---

## Data Description

The dataset consists of **customer demographics and interaction attributes** used to predict the likelihood of purchasing the Wellness Tourism Package.

### Target Variable

* **ProdTaken**: Indicates whether the customer purchased the package

  * `0` – No
  * `1` – Yes

---

### Customer Details

* **CustomerID**: Unique identifier for each customer
* **Age**: Age of the customer
* **TypeofContact**: Mode of contact (Company Invited / Self Inquiry)
* **CityTier**: City category based on development and population (Tier 1, Tier 2, Tier 3)
* **Occupation**: Customer’s profession (Salaried, Freelancer, etc.)
* **Gender**: Gender of the customer
* **NumberOfPersonVisiting**: Total number of people traveling with the customer
* **PreferredPropertyStar**: Preferred hotel star rating
* **MaritalStatus**: Marital status (Single, Married, Divorced)
* **NumberOfTrips**: Average number of annual trips
* **Passport**: Passport availability (0: No, 1: Yes)
* **OwnCar**: Car ownership (0: No, 1: Yes)
* **NumberOfChildrenVisiting**: Number of children below 5 years
* **Designation**: Job designation
* **MonthlyIncome**: Monthly income of the customer

---

### Customer Interaction Data

* **PitchSatisfactionScore**: Customer satisfaction score for the sales pitch
* **ProductPitched**: Product pitched to the customer
* **NumberOfFollowups**: Number of follow-ups after the sales pitch
* **DurationOfPitch**: Duration (in minutes) of the sales pitch

---

## Technology Stack

* **Programming Language**: Python
* **Machine Learning**: Scikit-learn
* **Version Control**: Git & GitHub
* **CI/CD Automation**: GitHub Actions
* **MLOps Practices**: Automated training, testing, and deployment readiness
