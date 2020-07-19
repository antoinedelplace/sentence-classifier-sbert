FROM python:3.6

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY api.py ./api.py
COPY clf.joblib ./clf.joblib

COPY model.zip ./model.zip
RUN unzip model.zip
RUN mkdir model
RUN mv content/model/* model/
RUN rm -r content
RUN rm model.zip