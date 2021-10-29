FROM python:3.7

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -U pip wheel cmake
RUN pip install -r requirements.txt

EXPOSE 8501

COPY . ./main 
# or could use ADD
WORKDIR ./main/

ENTRYPOINT ["streamlit", "run"]
CMD ["Streamlit_app.py"]