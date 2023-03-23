# Dockerfile -> Docker image -> Docker container
FROM python:3.8.13
RUN mkdir predict_customer_churn
RUN cd predict_customer_churn
WORKDIR predict_customer_churn
ADD . .
RUN pip install -r requirements.txt -q
CMD ["python", "./churn_library_logging.py"]