---
title: Experiments - Helpdesk event log
description: Code và kết quả các thí nghiệm của báo cáo nằm trong một notebook, xếp theo thứ tự mục của outline.
created: 2026-10-29
updated: 2026-10-30
tags: [advance-erp, helpdesk, experiments]
status: draft
---

# Experiments

Toàn bộ logic và kết quả: `helpdesk.ipynb`, xếp theo mục của `docs/report.md`.

| Phần trong notebook | Câu hỏi | Mục báo cáo |
|---|---|---|
| 5.1 | Ticket có kẹt hẳn ở Wait không? Wait làm chậm bao nhiêu? | 5.1 |
| 5.2–5.3 | Năm 2013 quy trình có đổi không? (Graph2Vec) | 5.2, 5.3 |
| 5.4 | Thời gian theo quý: từ quý 4/2012 mọi ticket chậm hơn | 5.4 |
| 6 | BPMN TO-BE có deadlock không? Nhắc sau X ngày thì nhanh hơn bao nhiêu? | 6 |
| 7 | Chia dữ liệu học / kiểm tra | 7.1 |
| 7.2 | Ticket nào hay rẽ vào Wait? (Decision Tree) | 7.2 |
| 7.3 | Đoán bước tiếp theo và thời gian còn lại đến Resolve ticket (3 seed); học và kiểm tra khác giai đoạn | 7.1, 7.3 |
| 7.4 | Tại điểm rẽ có đoán được Wait không | 7.4 |
| 7.5 | Hẹn nhắc theo số ngày dự đoán so với hẹn cố định X ngày | 7.5 |

Mục 1–4 (BPMN, Inductive Miner, alignment, Node2Vec) chạy trực tiếp trong app, code ở `src/helpdesk_app/process_model.py` và `bpmn_embedding.py`.

Code các mô hình dự đoán nằm ở `src/helpdesk_app/prediction.py`, dùng chung với app. Mở notebook để xem kết quả đã lưu. Chạy lại toàn bộ (khoảng 5 phút, phần lớn là LSTM và Transformer) từ thư mục `helpdesk_app/`:

```
uv run jupyter nbconvert --to notebook --execute --inplace experiments/helpdesk.ipynb
```
