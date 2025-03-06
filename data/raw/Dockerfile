FROM python:3.13

# Install kaggle and download xrays
RUN pip install kaggle

COPY src /src

WORKDIR /src

RUN chmod +x download.sh

CMD [ "./download.sh" ]