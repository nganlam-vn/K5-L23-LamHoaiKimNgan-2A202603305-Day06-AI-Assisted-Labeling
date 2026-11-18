# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 523; đối tượng sau duyệt: 517; ngưỡng accepted: IoU >= 0.9
- Đã **loại 18 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 485 | 485/523 = 92.7% | 523 box máy |
| edited (sửa) | 24 | 24/523 = 4.6% | 523 box máy |
| deleted (xoá) | 14 | 14/523 = 2.7% | 523 box máy |
| added (vẽ thêm) | 8 | 8/517 = 1.5% | 517 đối tượng sau duyệt |

Trong số deleted + added, **0 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 8.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 253 | 12 | 6 | 5 |
| bicycle | 7 | 1 | 0 | 0 |
| motorbike | 125 | 5 | 8 | 3 |
| car | 53 | 2 | 0 | 0 |
| bus | 2 | 1 | 0 | 0 |
| truck | 6 | 0 | 0 | 0 |
| three_wheeler | 39 | 3 | 0 | 0 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 45 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 33 | 33 | 12 | 50.0% (KTC 95%: 38–62%) | 73.3% (KTC 95%: 59–84%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 11/12 = 91.7% (KTC 95%: 65–99%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - extra: 3/3
  - wrong_class: 3/3
  - missing: 3/3
  - loose: 2/3
  - lọt: img_018.jpg (loose)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
