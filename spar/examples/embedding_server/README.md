# macOS External Embedding Server

별도 macOS 프로세스에서 `sentence-transformers` 기반 임베딩 서버를 띄우는 예제입니다.
SPAR 본체와 분리된 가상환경에서 실행하는 것을 전제로 합니다.

## 1. 가상환경 준비

```bash
cd examples/embedding_server
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. 서버 실행

기본 모델은 `BAAI/bge-large-en-v1.5`, 기본 디바이스는 `cpu`입니다.

```bash
export EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
export EMBEDDING_DEVICE=cpu
export EMBEDDING_API_KEY=dummy

uvicorn app:app --host 0.0.0.0 --port 9000
```

로컬 모델 디렉터리를 직접 지정해도 됩니다.

```bash
export EMBEDDING_MODEL=/absolute/path/to/models/bge-large-en-v1.5
uvicorn app:app --host 0.0.0.0 --port 9000
```

## 3. 서버 확인

```bash
curl http://127.0.0.1:9000/health

curl http://127.0.0.1:9000/v1/models

curl http://127.0.0.1:9000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer dummy' \
  -d '{
    "model": "BAAI/bge-large-en-v1.5",
    "input": ["Samsung LTE parameter reference example"]
  }'
```

## 4. SPAR 연결

SPAR 저장소 쪽에서는 아래처럼 설정합니다.

```bash
export EMBEDDING_URL=http://127.0.0.1:9000/v1
export ENCODER_URL=http://127.0.0.1:9000/v1
export ENCODER_MODEL=BAAI/bge-large-en-v1.5
export EMBEDDING_API_KEY=dummy

make test-embedding-server
```

그 다음 ingest/app 경로는 원격 임베딩 서버를 사용합니다.
