FROM python:3.11.7-slim
WORKDIR /src

# インストール関係
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# コンテナ起動時実行(未定)
# CMD ["python", "app.py"]