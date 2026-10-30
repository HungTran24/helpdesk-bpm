# helpdesk-bpm

Áp dụng BPM vào quy trình xử lý ticket Helpdesk: Process Mining, Petri Net, Node2Vec/Graph2Vec, BPMN TO-BE và mô hình dự đoán cho ticket đang chạy. Đồ án môn Hoạch định nguồn lực doanh nghiệp nâng cao, UIT.

Kết luận: chỗ chậm nhất là bước `Wait`. Cách sửa là tự động nhắc khi ticket chờ quá X ngày, với một X chung cho mọi ticket, vì ngày nhắc theo dự đoán của mô hình không tốt hơn.

## Cài đặt

Cần [uv](https://docs.astral.sh/uv/). Python 3.13 và các thư viện được cố định trong `.python-version` và `uv.lock`; PyTorch dùng bản CPU.

```bash
uv sync
```

## Chạy

```bash
uv run streamlit run src/helpdesk_app/app.py      # app, mở http://localhost:8501
uv run jupyter nbconvert --to notebook --execute --inplace experiments/helpdesk.ipynb   # chạy lại thí nghiệm, khoảng 5 phút
uv run python -m helpdesk_app.prediction          # học lại mô hình cho app và in điểm, khoảng 2 phút
uv run python -m helpdesk_app.bpmn_embedding      # Node2Vec + K-Means, 5 seed
```

App đọc mô hình đã học ở `models/predictors.pt`; nếu xoá file này, app tự học lại ở lần mở đầu tiên.

## Cấu trúc

```
data/dataset_Helpdesk_eventlog.csv   event log, không chỉnh sửa
src/helpdesk_app/
  event_log.py        đọc CSV, giữ tên cột gốc
  process_model.py    BPMN chuẩn, Inductive Miner, alignment, BPMN TO-BE
  bpmn_embedding.py   Node2Vec, K-Means, Graph2Vec
  waiting.py          thời gian Wait, thời gian theo quý, mô phỏng nhắc sau X ngày
  prediction.py       chia dữ liệu, Decision Tree, baseline, Gradient Boosting, LSTM, Transformer
  app.py              app Streamlit 3 màn hình
experiments/helpdesk.ipynb   thí nghiệm và kết quả đã lưu
models/predictors.pt         mô hình đã học cho app
```

App và notebook dùng chung các module trong `src/helpdesk_app/`.

## Tái hiện kết quả

Notebook lưu sẵn kết quả của lần chạy dùng trong báo cáo. Node2Vec, LSTM và Transformer xê dịch nhẹ giữa các lần chạy (hàm băm của Python trong gensim, PyTorch học nhiều luồng CPU), nên Node2Vec chạy 5 seed và các mô hình dự đoán chạy 3 seed, báo trung bình ± độ lệch chuẩn.

## Dữ liệu

Polato, M. (2017). *Dataset belonging to the help desk log of an Italian Company* (Version 1). University of Padova, 4TU.ResearchData. https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb

4.580 ticket, 21.348 sự kiện, 2010–2014. Phân phối lại theo 4TU General Terms of Use; xem bản gốc tại DOI trên.
